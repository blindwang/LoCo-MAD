import logging
from collections import OrderedDict, Counter
from itertools import repeat
import collections.abc
import math
import numpy as np
from tabulate import tabulate

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, reduce

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from timm.models.layers import DropPath

# from lavis.models.eva_vit import convert_weights_to_fp16
from lavis.common.dist_utils import download_cached_file


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self, d_model, down_ratio=0.25):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = int(d_model * down_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("d_fc1", nn.Linear(d_model, self.d_hidden)),
            ("gelu", QuickGELU()),
            ("d_fc2", nn.Linear(self.d_hidden, d_model))
        ]))
        # nn.init.zeros_(self.mlp.d_fc2.weight)
        # nn.init.zeros_(self.mlp.d_fc2.bias)

    def forward(self, x):
        """
        @param x: Tensor of Shape[(b t) (h*w+1) c]
        @return: Tensor of Shape[(b t) (h*w+1) c]
        """
        return self.mlp(x)


# ============================== Different Residual Attention Block ========================================
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, use_grad_checkpointing=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    @torch.autocast(enabled=False, device_type='cuda')  # for FlashAttention
    @torch.no_grad()
    def attention(self, x: torch.Tensor):
        self.attn.eval()
        # dtype = x.dtype
        x = x.half()
        return self.attn(x, x, x, need_weights=False)[0].float()

    def forward(self, x: torch.Tensor):
        t = x.shape[1]
        x = rearrange(x, 'b t k c -> (b t) k c')
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = rearrange(x, '(b t) k c ->b t k c', t=t)
        return x


class AIMResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_frames: int = 8, drop_path_rate: float = 0.1,
                 temporal_adapter=True, spatial_adapter=True, mlp_adapter=True, down_ratio=0.25, scale=0.5,
                 use_grad_checkpointing=False):
        super().__init__()
        self.num_frames = num_frames
        self.has_temporal_adapter = temporal_adapter
        self.has_spatial_adapter = spatial_adapter
        self.has_mlp_adapter = mlp_adapter
        self.scale = scale

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

        if temporal_adapter is True:
            self.t_adapter = Adapter(d_model, down_ratio=down_ratio)
        if spatial_adapter is True:
            self.s_adapter = Adapter(d_model, down_ratio=down_ratio)
        if mlp_adapter is True:
            self.mlp_adapter = Adapter(d_model, down_ratio=down_ratio)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    @torch.autocast(enabled=False, device_type='cuda')  # for FlashAttention
    @torch.no_grad()
    def attention(self, x: torch.Tensor):
        self.attn.eval()
        # dtype = x.dtype
        x = x.half()
        return self.attn(x, x, x, need_weights=False)[0].float()

    def forward(self, x: torch.Tensor):
        """
        @param x: Tensor of Shape[b t (hw+1) c]
        @return:
        """
        b, t, k, c = x.shape

        # temporal
        if self.has_temporal_adapter:
            x = rearrange(x, 'b t k c -> (b k) t c')
            x = x + self.drop_path(self.t_adapter(self.attention(self.ln_1(x))))
            x = rearrange(x, '(b k) t c -> b t k c', b=b, k=k)
        # spatial
        x = rearrange(x, 'b t k c -> (b t) k c')
        if self.has_spatial_adapter:
            x = x + self.s_adapter(self.attention(self.ln_1(x)))
        else:
            x = x + self.attention(self.ln_1(x))
        x = rearrange(x, '(b t) k c -> b t k c', b=b, t=t)
        # MLP
        mlp_x = self.ln_2(x)
        if self.has_mlp_adapter:
            x = x + self.mlp(mlp_x) + self.drop_path(self.scale * self.mlp_adapter(mlp_x))
        else:
            x = x + self.mlp(mlp_x)
        return x


class MsgExchangeResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 use_grad_checkpointing=False, t: int = 8):
        super().__init__()
        # temporal dimension
        self.t = t
        # message
        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head)

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # x: L N(b t) D
        l, bt, d = x.size()
        b = bt // self.t

        # message
        msg = self.message_ln(self.message_fc(x[0, :, :]))  # (b t) d
        msg = rearrange(msg, '(b t) d -> t b d', b=b, t=self.t)
        msg += self.message_attn(msg, msg, msg, need_weights=False)[0]
        msg = rearrange(msg, 't b d -> 1 (b t) d')
        x = torch.cat([x, msg], dim=0)  # l+1 (b t) d

        x = x + self.attention(self.ln_1(x))
        x = x[:l, :, :]
        x = x + self.mlp(self.ln_2(x))
        return x


class AdapterResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 use_grad_checkpointing=False, t: int = 8, window_size=(4, 4, 4)):
        super().__init__()
        # temporal dimension
        self.t = t
        # adapter
        self.perceiver_adapter = nn.Sequential(OrderedDict([
            ("d_fc", nn.Linear(d_model, d_model // 4)),
            ("gelu", QuickGELU()),
            ("u_fc", nn.Linear(d_model // 4, d_model))
        ]))
        self.perceiver_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.window_size = window_size

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def window_partition(self, x):
        """
        Args:
            x: (B, T, H, W, C)
        Returns:
            windows: (B*num_windows, window_size*window_size, C)
        """
        B, D, H, W, C = x.shape
        x = x.view(
            B,
            D // self.window_size[0], self.window_size[0],
            H // self.window_size[1], self.window_size[1],
            W // self.window_size[2], self.window_size[2],
            C
        )
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, math.prod(self.window_size), C)
        return windows

    def cross_attention(self, x: torch.Tensor, query: torch.Tensor):
        """
        @param x: Tensor of Shape[L N(b t) C]
        @param query: Tensor of Shape[b, num_q, C]
        @return: query: Tensor of Shape[b, num_q, C]
        """
        l, bt, c = x.size()
        b = bt // self.t
        h = w = int(math.sqrt(l - 1))
        num_q = h * w * self.t // int(math.prod(self.window_size))

        window_x = rearrange(x[1:, :, :], '(h w) (b t) c -> b t h w c', b=b, t=self.t, h=h, w=w)
        window_x = self.window_partition(window_x)  # B*num_q, wh*ww*wt, C
        query = rearrange(query, 'b q c -> (b q) 1 c')  # B*num_q, 1, C
        query = self.perceiver_attn(query, window_x, window_x, need_weights=False)[0]  # B*num_q, 1, C
        return rearrange(query, '(b q) 1 c -> b q c', b=b, q=num_q)

    def forward(self, x: torch.Tensor, query: torch.Tensor):
        """
        @param x: Tensor of Shape[L N(b t) C]
        @param query: Tensor of Shape[b, num_q, C]
        @return:
        """
        q = self.cross_attention(x, query)  # b, num_q, c
        q = self.attention(self.ln_1(q.permute(1, 0, 2))).permute(1, 0, 2)
        q = query + self.perceiver_adapter(q)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x, q


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 use_grad_checkpointing=False, t: int = 8, msg_exchange_layers: tuple = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            MsgExchangeResidualAttentionBlock(width, heads, attn_mask, use_grad_checkpointing and i > 12, t=t)
            if i in msg_exchange_layers else
            ResidualAttentionBlock(width, heads, attn_mask, use_grad_checkpointing and i > 12, t=t)
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class DualPathTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 use_grad_checkpointing=False, t: int = 8, num_query: int = 32,
                 adapted_layers: tuple = None, window_size: tuple = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_query = num_query
        self.perceiver_query = nn.Parameter(torch.zeros(num_query, width))
        self.perceiver_query.data.normal_(mean=0.0, std=0.02)
        self.t = t

        kwargs = {'d_model': width, 'n_head': heads, 'attn_mask': attn_mask, 't': t}
        self.resblocks = nn.ModuleList([
            AdapterResidualAttentionBlock(window_size=window_size, **kwargs)
            if i in adapted_layers else
            ResidualAttentionBlock(**kwargs)
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        query = self.perceiver_query.unsqueeze(0).expand(x.shape[1] // self.t, -1, -1)
        for l in self.resblocks:
            if type(l) is AdapterResidualAttentionBlock:
                x, query = l(x, query)
            elif type(l) is ResidualAttentionBlock:
                x = l(x)
            else:
                raise AssertionError("WRONG BLOCK")
        return x, query


class AIMTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, num_frames: int = 8, drop_path_rate: float = 0.1,
                 temporal_adapter=True, spatial_adapter=True, mlp_adapter=True,
                 down_ratio=0.25, scale=0.5,
                 use_grad_checkpointing=False, adapted_layers: tuple = ()):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_frames = num_frames
        kwargs = dict(num_frames=num_frames, drop_path_rate=drop_path_rate, temporal_adapter=temporal_adapter,
                      spatial_adapter=spatial_adapter, mlp_adapter=mlp_adapter, down_ratio=down_ratio, scale=scale)
        self.resblocks = nn.Sequential(*[
            AIMResidualAttentionBlock(width, heads, **kwargs)
            if i in adapted_layers else
            ResidualAttentionBlock(width, heads, use_grad_checkpointing and i > 12)
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        x = rearrange(x, '(b t) k c -> b t k c', t=self.num_frames)
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 use_grad_checkpointing: bool, t: int = 8, msg_exchange_layers: tuple = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        self.num_patches = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_grad_checkpointing=use_grad_checkpointing,
                                       t=t, msg_exchange_layers=msg_exchange_layers)

        # self.ln_final = LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_final(x)
        return x


class DualPathVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 use_grad_checkpointing: bool, t: int = 8, adapted_layers: tuple = None,
                 window_size: tuple = (4, 4, 4)):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        self.num_patches = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = DualPathTransformer(
            width, layers, heads, use_grad_checkpointing=use_grad_checkpointing,
            t=t, adapted_layers=adapted_layers, window_size=window_size,
            num_query=self.num_patches * t // int(math.prod(window_size))
        )

        # self.ln_final = LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, q = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_final(x)
        return x, q


class AIMVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 use_grad_checkpointing: bool, num_frames: int = 8, adapted_layers: tuple = (),
                 drop_path_rate: float = 0.1, temporal_adapter=True, spatial_adapter=True, mlp_adapter=True,
                 down_ratio=0.25, scale=0.5,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        self.num_patches = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(self.num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = AIMTransformer(
            width, layers, heads,
            use_grad_checkpointing=use_grad_checkpointing,
            num_frames=num_frames,
            drop_path_rate=drop_path_rate,
            temporal_adapter=temporal_adapter,
            spatial_adapter=spatial_adapter,
            mlp_adapter=mlp_adapter,
            down_ratio=down_ratio,
            scale=scale,
            adapted_layers=adapted_layers,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)
        return x

    def freeze_backbone(self):
        for k, v in self.named_parameters():
            if 't_adapter' in k or 's_adapter' in k or 'mlp_adapter' in k:
                continue
            v.requires_grad = False
        logging.info("Freezing AIM vit backbone")


def parse_adapted_mode(adapted_mode, num_layers):
    if adapted_mode is None:
        return tuple(range(num_layers))
    mode, num = adapted_mode.split('_')
    num = int(num)
    if mode == 'last':
        layer_idx = tuple(range(num_layers - num, num_layers))
    elif mode == 'every':
        layer_idx = tuple(range(num_layers))[::num]
    elif mode == 'skip1every':
        layer_idx = tuple(range(num_layers))[1::num]
    else:
        raise ValueError
    return layer_idx


def prettify_state_key_match(incompatible_keys):
    missing_counter = Counter(['.'.join(i.split('.')[:4]) for i in incompatible_keys.missing_keys])
    print(tabulate([('missing', '#')] + list(missing_counter.items())))
    unexpected_counter = Counter(['.'.join(i.split('.')[:3]) for i in incompatible_keys.unexpected_keys])
    print(tabulate([('unexpected', '#')] + list(unexpected_counter.items())))


def convert_attn_weights_to_fp32(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

    model.apply(_convert_weights_to_fp32)


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        # if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        #     l.weight.data = l.weight.data.half()
        #     if l.bias is not None:
        #         l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            # for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
            #     tensor = getattr(l, attr)
            #     if tensor is not None:
            #         tensor.data = tensor.data.half()
            l.half()

    model.apply(_convert_weights_to_fp16)


def create_adapted_vitl(img_size=224, use_checkpoint=False, precision="fp16", jit=True, adapted_mode='last_4'):
    msg_exchange_layers = parse_adapted_mode(adapted_mode, 22)

    model = VisionTransformer(
        input_resolution=img_size,
        patch_size=14,
        width=1024,
        layers=22,
        heads=16,
        use_grad_checkpointing=use_checkpoint,
        msg_exchange_layers=msg_exchange_layers,
    )
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")
    # interpolate_pos_embed(model, state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print(f"CLIP {incompatible_keys}")

    if precision == "fp16":
        convert_weights_to_fp16(model)

    if jit is True:
        # with torch.autocast(device_type='cuda'):
        #     with torch.no_grad():
        #         model = torch.jit.trace(model, torch.rand(16, 3, 224, 224, device='cuda'))
        model = torch.compile(model)
    return model


def create_adapted_vitl_v2(img_size=224, use_checkpoint=False, precision="fp16", jit=True, adapted_mode='skip1every_2'):
    adapted_layers = parse_adapted_mode(adapted_mode, 22)

    model = DualPathVisionTransformer(
        input_resolution=img_size,
        patch_size=14,
        width=1024,
        layers=22,
        heads=16,
        use_grad_checkpointing=use_checkpoint,
        adapted_layers=adapted_layers,
    )
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")
    # interpolate_pos_embed(model, state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print(f"CLIP {incompatible_keys}")

    if precision == "fp16":
        convert_weights_to_fp16(model)

    if jit is True:
        # with torch.autocast(device_type='cuda'):
        #     with torch.no_grad():
        #         model = torch.jit.trace(model, torch.rand(16, 3, 224, 224, device='cuda'))
        model = torch.compile(model)
    return model


def create_aim_adapted_vitl(
        img_size=224, use_checkpoint=False, adapted_mode='skip1every_2',
        drop_path_rate=0.1, temporal_adapter=True, spatial_adapter=True, mlp_adapter=True, down_ratio=0.25, scale=0.5,
        freeze_vit=True,
        **kwargs
):
    adapted_layers = parse_adapted_mode(adapted_mode, 22)
    model = AIMVisionTransformer(
        input_resolution=img_size,
        patch_size=14,
        width=1024,
        layers=22,
        heads=16,
        use_grad_checkpointing=use_checkpoint,
        adapted_layers=adapted_layers,
        drop_path_rate=drop_path_rate,
        temporal_adapter=temporal_adapter, spatial_adapter=spatial_adapter, mlp_adapter=mlp_adapter,
        down_ratio=down_ratio, scale=scale,
    )

    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth"
    cached_file = download_cached_file(url, check_hash=False, progress=True)
    state_dict = torch.load(cached_file, map_location="cpu")

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print("ViT:")
    prettify_state_key_match(incompatible_keys)
    convert_weights_to_fp16(model)
    # convert_attn_weights_to_fp32(model)

    model.num_features = 1024
    if freeze_vit:
        model.freeze_backbone()
    return model
