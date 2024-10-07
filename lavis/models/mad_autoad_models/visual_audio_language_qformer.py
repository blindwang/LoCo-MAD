"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import re
from collections import Counter, defaultdict
from tabulate import tabulate
import time
import datetime
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
import numpy as np
from typing import List, Optional

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather, BaseModel
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    # compute_sim_matrix,
    disabled_train,
    BertConfig,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from lavis.common.dist_utils import download_cached_file
import lavis.common.dist_utils as dist_utils
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from .Qformer import (
    QformerLMHeadModel
)
# from .temporal_encoder import MessageExchangeEncoder, create_xclip_cct_model, BiAttentionAdapter
from .vit_adapter import create_adapted_vitl, create_adapted_vitl_v2, create_aim_adapted_vitl
import language_evaluation.coco_caption_py3.pycocoevalcap as eval_tools

from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BertTokenizer,
    AutoTokenizer,
)
from transformers.modeling_utils import apply_chunking_to_forward


class VanillaPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 100):
        super(VanillaPositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.pos_embedding[:token_embedding.shape[-2], :]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class AttentionPooler(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads):
        super().__init__()
        self.num_attention_heads = n_heads
        self.attention_head_size = int(output_dim / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = nn.Linear(output_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, pooler_query):
        """
        Args:
            x: Tensor[B, L, D], the feature of one modality.
            pooler_query: Tensor[1, 1, E]
        Returns:
            Tensor[B, 1, E], the pooled global feature.
        """
        bs = x.shape[0]
        # [B, L, D] -> [B, L, E] -> [B, h, L, d]
        key_layer = self.transpose_for_scores(self.key(x))
        value_layer = self.transpose_for_scores(self.value(x))
        # [1, 1, E] -> [B, 1, E] -> [B, h, 1, d]
        query_layer = self.transpose_for_scores(pooler_query.expand(bs, -1, -1))
        # [B, h, 1, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # [B, h, 1, L] x [B, h, L, d] -> [B, h, 1, d] -> [B, 1, E]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


def top_k_intra_NCE_loss(x, y, top_k, sim_temp):
    """
    Args:
        sim_temp: similarity temperature
        top_k: select top_k positives
        x: Tensor[B, n1, E]
        y: Tensor[B, n2, E]
    """
    # b, n1, n2 = x.shape[0], x.shape[1], y.shape[1]
    # sim = torch.einsum('bpc,bqc->bpq', x, y).view(b, -1)  # [B, n1, n2] -> [B, n1xn2]
    # sim /= sim_temp
    # th_value = sim.topk(top_k, dim=-1).values[:, -1].unsqueeze(-1)  # [b, 1]
    # pos = torch.logsumexp(sim[sim >= th_value][:b * top_k].view(b, top_k),
    #                       dim=1)  # [B, top_k] -> exp -> sum -> [B, 1] -> log
    # pos_neg = torch.logsumexp(sim, dim=1)  # [B, top_k] -> exp -> sum -> [B, 1] -> log
    # return -torch.mean(pos - pos_neg)
    # sim_v2t[sim_v2t < th_value] *= 1e-4
    # sim_v2t = sim_v2t.mean(-1)
    # sim_v2t, _ = sim_q2t.max(-1)  # [batch_size, batch_size*num_gpu]
    b, n1, n2 = x.shape[0], x.shape[1], y.shape[1]
    sim = torch.einsum('bpc,bqc->bpq', x, y).view(b, -1)  # [B, n1, n2] -> [B, n1xn2]
    sim /= sim_temp
    th_value = sim.topk(top_k, dim=-1).values[:, -1].unsqueeze(-1)  # [b, 1]
    pos_sim = sim.clone()
    pos_sim[sim < th_value] = -10e3
    pos = torch.logsumexp(pos_sim, dim=1)  # [B, top_k] -> exp -> sum -> [B, 1] -> log
    pos_neg = torch.logsumexp(sim, dim=1)  # [B, top_k] -> exp -> sum -> [B, 1] -> log
    return -torch.mean(pos - pos_neg)


def top_k_batch_NCE_loss(x, y, top_k, sim_temp):
    """
    Args:
        x: Tensor(B, n1, E)
        y: Tensor(B, n2, E)
        sim_temp: similarity temperature
        top_k: select top_k positives
    """
    b, n1, n2 = x.shape[0], x.shape[1], y.shape[1]
    device = x.device
    rank: int = dist.get_rank()
    ws: int = dist_utils.get_world_size()
    targets = torch.linspace(0, b - 1, b).to(device, dtype=int)

    # calculate mean intra sim mat [B, B, n1, n2] -> [B, B, n1xn2]
    sim_x_y = torch.einsum('bpc,nqc->bnpq', x, y).reshape(b, b, -1)
    th_value = sim_x_y.topk(top_k, dim=-1).values[:, :, -1].unsqueeze(-1)
    sim_x_y[sim_x_y < th_value] *= 1e-4
    sim_x_y = sim_x_y.mean(-1) / sim_temp  # [B, B]

    if dist_utils.is_dist_avail_and_initialized() and ws != 1:
        x, y = x.contiguous(), y.contiguous()
        # get from other rank
        x_gather = [torch.ones_like(x) for _ in range(dist_utils.get_world_size())]
        torch.distributed.all_gather(x_gather, x, async_op=False)
        del x_gather[rank]
        other_x = torch.cat(x_gather, dim=0)

        y_gather = [torch.ones_like(y) for _ in range(dist_utils.get_world_size())]
        torch.distributed.all_gather(y_gather, y, async_op=False)
        del y_gather[rank]
        other_y = torch.cat(y_gather, dim=0)

        # calculate x <-> other_y [B, N, n1, n2] -> [B, N, n1xn2]
        sim_x_other = torch.einsum('bpc,nqc->bnpq', x, other_y).reshape(b, other_y.shape[0], -1)
        th_value = sim_x_other.topk(top_k, dim=-1).values[:, :, -1].unsqueeze(-1)
        sim_x_other[sim_x_other < th_value] *= 1e-4
        sim_x_other = sim_x_other.mean(-1) / sim_temp  # [B, N]

        # calculate other_x <-> y [N, B, n1, n2] -> [N, B, n1xn2]
        sim_other_y = torch.einsum('npc,bqc->nbpq', other_x, y).reshape(other_x.shape[0], b, -1)
        th_value = sim_other_y.topk(top_k, dim=-1).values[:, :, -1].unsqueeze(-1)
        sim_other_y[sim_other_y < th_value] *= 1e-4
        sim_other_y = sim_other_y.mean(-1) / sim_temp  # [N, B]

        sim_x2y = torch.cat([sim_x_y, sim_x_other], dim=1)
        sim_y2x = torch.cat([sim_x_y.T, sim_other_y], dim=1)
        loss_x2y = F.cross_entropy(sim_x2y, targets, label_smoothing=0.1)
        loss_y2x = F.cross_entropy(sim_y2x, targets, label_smoothing=0.1)
        return (loss_x2y + loss_y2x) / 2, sim_x2y, sim_y2x
    else:
        loss_x2y = F.cross_entropy(sim_x_y, targets, label_smoothing=0.1)
        loss_y2x = F.cross_entropy(sim_x_y.T, targets, label_smoothing=0.1)
        return (loss_x2y + loss_y2x) / 2, sim_x_y, sim_x_y.T


class QformerBase(BaseModel):
    @classmethod
    def init_tokenizer(cls, lang='en'):
        if lang == 'en':
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif lang == 'ch':
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        else:
            raise ValueError
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, vocab_size=30523,
                     audio=False, audio_max_len=None,
                     local_query_windows_size=None, num_local_query=0, num_frames=0, pyramid_temp_query=False,
                     token_drop_rate=0.0):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.vocab_size = vocab_size
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.audio = audio
        encoder_config.audio_max_len = audio_max_len
        encoder_config.disagree_regularize_config = None
        encoder_config.local_query_windows_size = local_query_windows_size
        encoder_config.num_local_query = num_local_query
        encoder_config.num_frames = num_frames
        encoder_config.token_drop_rate = token_drop_rate
        encoder_config.pyramid_temp_query = pyramid_temp_query

        Qformer = QformerLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config,
            ignore_mismatched_sizes=True  # For Chinese Training
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
            cls, vit_model, img_size, precision="fp32", use_grad_checkpoint=False, drop_path_rate=0.1, **kwargs
    ):
        assert vit_model in [
            "eva_clip_g", "clip_L", "aim"
        ], "vit model must be eva_clip_g or clip_L"
        if vit_model == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        elif vit_model == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        elif vit_model == "aim":
            visual_encoder = create_aim_adapted_vitl(
                img_size, use_grad_checkpoint, drop_path_rate=drop_path_rate,
                **kwargs
            )
        else:
            raise AssertionError('vit model must be eva_clip_g or clip_L')
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    @classmethod
    def init_adapted_vision_encoder(
            cls, img_size, precision, jit=True, adapted_mode=None
    ):
        visual_encoder = create_adapted_vitl_v2(
            img_size=img_size,
            precision=precision,
            jit=jit,
            adapted_mode=adapted_mode
        )
        visual_encoder.num_features = 1024  # set attribute
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    @classmethod
    def init_audio_encoder(cls):
        from transformers import ASTModel
        audio_encoder = ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
        ln_audio = LayerNorm(audio_encoder.config.hidden_size)
        return audio_encoder, ln_audio

    def _suit_query_token(self, state_dict):
        if 'query_tokens' in state_dict:
            tgt = self.query_tokens.shape[1]
            src = state_dict['query_tokens'].shape[1]
            if tgt != src:
                # logging.info(f"reset query_token to {tgt}")
                # del state_dict['query_tokens']
                if tgt % src == 0:
                    state_dict['query_tokens'] = state_dict['query_tokens'].expand(tgt // src, -1, -1).reshape(1, tgt, -1)
                else:
                    state_dict['query_tokens'] = torch.cat([
                        state_dict['query_tokens'], self.query_tokens[:, src:, :]
                    ], dim=1)
                logging.info(f"transfer query_token from {src} to {tgt}")

    def _suit_temporal_embedding(self, state_dict: dict):
        te = [v for k, v in state_dict.items() if 'visual_temporal_embeddings' in k]
        if len(te) == self.visual_num_temporal_embedding:
            return state_dict
        elif len(te) == 0:
            return state_dict
        else:
            logging.info(f"interpolate temporal embedding from {len(te)} to {self.visual_num_temporal_embedding}")
            te = torch.stack(te, dim=1).unsqueeze(0)  # 1, 1024, 8
            # 1, 1024, 16
            new_te = torch.nn.functional.interpolate(te, size=(self.visual_num_temporal_embedding,), mode='linear')
            for i in range(self.visual_num_temporal_embedding):
                state_dict[f'visual_temporal_embeddings.{i}'] = new_te[0, :, i]
            return state_dict

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        if self.reset_pretrain_queries:
            logging.info(f"reset query_token")
            del state_dict['query_tokens']
        else:
            self._suit_query_token(state_dict)
        self._suit_temporal_embedding(state_dict)
        msg = self.load_state_dict(state_dict, strict=False)

        missing_counter = Counter(['.'.join(i.split('.')[:5]) for i in msg.missing_keys])
        missing_keys = tabulate([('missing', '#'), ] + list(missing_counter.items()))
        logging.info("\nMissing keys \n{}".format(missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


@registry.register_model("mad_qformer")
class MADQformer(QformerBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/VALQformer/base_modeling.yaml",
        "audio": "configs/models/VALQformer/audio_modeling.yaml",
        "end2end": "configs/models/VALQformer/base_end2end_modeling.yaml",
        "end2end_vitl": "configs/models/VALQformer/vitl_end2end_inference_modeling.yaml",
        "end2end_vitl_ch": "configs/models/VALQformer/vitl_ch_end2end_inference_modeling.yaml",
        "end2end_vitl_te": "configs/models/VALQformer/vitl_te_end2end_inference_modeling.yaml",
        "end2end_vitl_audio": "configs/models/VALQformer/vitl_audio_end2end_inference_modeling.yaml",
        "end2end_vitl_audio_q64": "configs/models/VALQformer/vitl_audio_q64_end2end_inference_modeling.yaml",
    }

    def __init__(
            self,
            # loss selection
            loss_config: dict[str: dict] = None,
            # original configurations
            num_query_token=32,
            embed_dim=256,
            max_txt_len=32,
            cross_attention_freq=2,
            vocab_size=30523,
            # My dev: feature
            encoder_config: dict = None,
            # My dev: addition
            visual_num_temporal_embedding=None,
            visual_num_spatial_embedding=None,
            query_temporal_embedding_config=None,
            visual_temporal_encoder_config=None,
            visual_local_query_config=None,
            discard_visual_cls_token=False,
            audio=True,
            audio_dim_features=768,
            audio_max_len=8,
            reset_pretrain_queries=False,
            token_drop_rate=0.0,
            # My dev: freeze
            freeze_modules: Optional[List[str]] = None,
    ):
        """

        Args:
            loss_config: dict, contains loss configurations. {type: {weight: int, **others}}
            num_query_token: int
            embed_dim: int, dimension of low dim visual and text representation
            max_txt_len: int, max text length
            cross_attention_freq: int, default 2
            vocab_size: int
            encoder_config: dict of encoder. [{end2end: bool, dim_features: int, **others}]
            visual_num_temporal_embedding: int, equal to number of frames
            query_temporal_embedding_config: dict, {num_query_te: int}
            visual_temporal_encoder_config: dict, deprecated
            visual_local_query_config: dict, {local_query_windows_size: list, num_local_query: int, num_frames: int}
            discard_visual_cls_token: bool
            audio: bool, using audio modality
            audio_dim_features: int
            audio_max_len: int
            reset_pretrain_queries: bool, re-init parameter of queries
            freeze_modules: list, 'self', 'cross', 'ffn', 'ffn_vi', 'ffn_txt'
        """
        super().__init__()
        # =========================================================================
        #                              Save Attributes
        # =========================================================================
        self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token
        self.embed_dim = embed_dim
        self.discard_visual_cls_token = discard_visual_cls_token
        self.audio = audio
        self.visual_num_temporal_embedding = visual_num_temporal_embedding
        self.visual_num_spatial_embedding = visual_num_spatial_embedding
        self.visual_temporal_encoder_config = visual_temporal_encoder_config
        self.visual_local_query_config = visual_local_query_config if visual_local_query_config is not None else {}
        self.query_temporal_embedding_config = query_temporal_embedding_config
        self.encoder_config = encoder_config
        self.end2end = encoder_config['end2end']
        self.freeze_modules = freeze_modules if freeze_modules is not None else []
        self.reset_pretrain_queries = reset_pretrain_queries
        self.loss_config = loss_config
        self._validate_loss_cfg()

        # =========================================================================
        #                            Input Configuration
        # =========================================================================
        # Selection of feature extraction
        # (audio_encoder), ln_audio, audio_proj, audio_max_len
        # (visual_encoder), ln_vision, dim_features
        if encoder_config['end2end'] is True:
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(**encoder_config)
            # self.visual_encoder, self.ln_vision = self.init_adapted_vision_encoder(
            #     img_size=encoder_config['img_size'],
            #     precision=encoder_config['vit_precision'],
            #     jit=encoder_config.get('jit', True),
            #     adapted_mode=encoder_config.get('adapted_mode', 'last_4'),
            # )
            self.dim_features = self.visual_encoder.num_features
            # if encoder_config.get('freeze_vit', True) is True:
            #     for name, param in self.visual_encoder.named_parameters():
            #         if "perceiver" in name:
            #             # print(name)
            #             continue
            #         param.requires_grad = False
            #     self.visual_encoder = self.visual_encoder.eval()
            #     self.visual_encoder.train = disabled_train
            #     logging.info("freeze vision encoder")
            if audio is True:
                self.audio_max_len = audio_max_len
                self.audio_encoder, self.ln_audio = self.init_audio_encoder()
                if self.audio_encoder.config.hidden_size != 768:
                    self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, 768)
                else:
                    self.audio_proj = nn.Identity()
                for p in self.audio_encoder.parameters():
                    p.requires_grad = False
        else:
            self.dim_features = encoder_config['dim_features']
            self.ln_vision = nn.LayerNorm(self.dim_features)
            # audio
            if audio:
                self.audio_max_len = audio_max_len
                if audio_dim_features != 768:  # Qformer is 768-d
                    self.audio_proj = nn.Linear(audio_dim_features, 768)
                else:
                    self.audio_proj = nn.Identity()
                self.ln_audio = nn.LayerNorm(audio_dim_features)
        # Init Tokenizer
        self.tokenizer = self.init_tokenizer(lang='en' if vocab_size == 30523 else 'ch')
        # Init Temporal Embedding
        if visual_num_temporal_embedding is not None:
            logging.info('activate num_temporal_embedding')
            self.pos_encoding = VanillaPositionalEncoding(self.dim_features, visual_num_temporal_embedding)
            # self.visual_temporal_embeddings = nn.ParameterList(
            #     nn.Parameter(torch.zeros(self.dim_features))
            #     for _ in range(visual_num_temporal_embedding)
            # )
        else:
            self.visual_temporal_embeddings = None
        if visual_num_spatial_embedding is not None:
            logging.info('activate num_spatial_embedding')
            self.visual_spatial_embeddings = nn.Embedding(16 * 16, self.dim_features)
        else:
            self.visual_spatial_embeddings = None
        # Init Visual Temporal Encoder
        if self.visual_temporal_encoder_config is not None:
            # self.visual_temporal_encoder = MessageExchangeEncoder(visual_temporal_encoder_config)
            # self.visual_temporal_encoder = create_xclip_cct_model(
            #     T=self.visual_num_temporal_embedding,
            #     start_layer_idx=visual_temporal_encoder_config.get('start_layer_idx', 21),
            #     end_layer_idx=visual_temporal_encoder_config.get('end_layer_idx', 21),
            # )
            # self.visual_temporal_encoder = BiAttentionAdapter(
            #     d_model=visual_temporal_encoder_config.get('hidden_size', 1024),
            #     num_temporal_embedding=self.visual_num_temporal_embedding
            # )
            pass
        # Init Query temporal embedding
        if self.query_temporal_embedding_config is not None:
            # ok with local query
            assert len(self.visual_local_query_config) > 0
            num_query_te = self.query_temporal_embedding_config['num_query_te']
            self.query_temporal_embeddings = nn.Parameter(torch.zeros(num_query_te, 768))
            logging.info(f"Add {num_query_te} to {self.num_query_token}: {self.num_query_token // num_query_te}/win")
        else:
            self.query_temporal_embeddings = None

        # =========================================================================
        #                          Backbone Configuration
        # =========================================================================
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.dim_features, vocab_size=vocab_size,
            audio=audio, audio_max_len=audio_max_len,
            cross_attention_freq=cross_attention_freq,
            token_drop_rate=token_drop_rate,
            **self.visual_local_query_config
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # =========================================================================
        #                          Output Configuration
        # =========================================================================
        if {'ITC', 'VAC', 'VATC', 'AVTC', 'topk-VTC'} & set(self.loss_config.keys()):
            self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.temp = nn.Parameter(0.07 * torch.ones([]))
            # self.temp = 0.02
        if {'topk-VATC', 'e-topk-VATC'} & set(self.loss_config.keys()):
            self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.audio_low_dim_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.vt_temp = nn.Parameter(0.05 * torch.ones([]))
            self.va_temp = nn.Parameter(0.05 * torch.ones([]))
            self.at_temp = nn.Parameter(0.05 * torch.ones([]))
        if {'att-VTC'} & set(self.loss_config.keys()):
            self.pooler = AttentionPooler(self.Qformer.config.hidden_size, embed_dim, n_heads=8)
            self.pooler_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pooler_query.data.normal_(mean=0.0, std=0.02)
        if {'ITM'} & set(self.loss_config.keys()):
            self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        if {'VAC', 'VATC', 'AVTC'} & set(self.loss_config.keys()):
            self.audio_low_dim_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.vac_temp = nn.Parameter(0.07 * torch.ones([]))
        # =========================================================================
        #                          Freeze Parameters
        # =========================================================================
        # If only do LM task or if training scst, the last layer of query is of no use
        # disable grad of the last non-language expert layers of encoder
        if ('SCST' in self.loss_config
                or 'VQA' in self.loss_config
                or ('LM' in self.loss_config and len(self.loss_config) == 1)):
            last_layer_num = 0
            for name, param in self.Qformer.named_parameters():
                l_num = re.findall(r"bert\.encoder\.layer\.([0-9]{1,2})\.", name)
                if len(l_num) > 0 and int(l_num[0]) > last_layer_num:
                    last_layer_num = int(l_num[0])
            logging.info(f"Disabling grad of the last non-language expert layers of encoder: {last_layer_num}")
            for name, param in self.Qformer.named_parameters():
                if f"bert.encoder.layer.{last_layer_num}" in name:
                    if "_query" in name:
                        param.requires_grad = False
                    elif audio is True and "_audio" in name:
                        param.requires_grad = False
        # If we don't do LM task, the cls.predictions layer is of not use.
        if 'LM' not in self.loss_config:
            logging.info(f"Disabling grad of language decoder layer")
            for param in self.Qformer.cls.predictions.parameters():
                param.requires_grad = False
        # Freeze pretrained modules
        for name, param in self.Qformer.named_parameters():
            if re.match(r'bert\.encoder\.layer\.[0-9]{1,2}\.[\w\.]*$', name):
                _sub_name = re.sub(r'bert\.encoder\.layer\.[0-9]{1,2}\.', '', name)
                if re.match('(intermediate_query\.)|(output_query\.)[\w\.]+', _sub_name):
                    if 'ffn' in self.freeze_modules or 'ffn_vi' in self.freeze_modules:
                        logging.info(f'Freeze: {name}')
                        param.requires_grad = False
                elif re.match('(intermediate\.)|(output\.)[\w\.]+', _sub_name):
                    if 'ffn' in self.freeze_modules or 'ffn_txt' in self.freeze_modules:
                        logging.info(f'Freeze: {name}')
                        param.requires_grad = False
                elif 'self' in self.freeze_modules and re.match('(attention\.)[\w\.]+', _sub_name):
                    logging.info(f'Freeze: {name}')
                    param.requires_grad = False
                elif 'cross' in self.freeze_modules and re.match('(crossattention\.)[\w\.]+', _sub_name):
                    logging.info(f'Freeze: {name}')
                    param.requires_grad = False

        # Ugly hack
        if 'SCST' in self.loss_config:
            self._set_grad_of_generate()

    def _validate_loss_cfg(self):
        self.loss_config: dict[dict]
        # assert type(self.loss_config) is dict
        assert len(self.loss_config) > 0
        loss_names = ("ITM", "ITC", "LM",
                      "VAC", "VATC", "AVTC", "att-VTC",
                      "topk-VTC", "topk-VATC", "e-topk-VATC",
                      "DAL",
                      "SCST", "VQA")

        logging.info("Calculating Loss: ")
        for k, params in self.loss_config.items():
            if k not in loss_names:
                raise ValueError(f"loss {k} not in {loss_names}")
            if 'weight' not in params:
                self.loss_config[k]['weight'] = 1.0
            logging.info(f"{k}: {params['weight']:.1f}")

        if 'SCST' in self.loss_config or 'VQA' in self.loss_config:
            assert 'LM' in self.loss_config and len(self.loss_config) <= 2

    @classmethod
    def from_config(cls, cfg):
        # cal_itm = cfg.get("cal_itm", True)
        # cal_itc = cfg.get("cal_itc", True)
        # cal_lm = cfg.get("cal_lm", True)
        # cal_vac = cfg.get("cal_vac", False)
        # cal_vatc = cfg.get("cal_vatc", False)
        # cal_vatc_v2 = cfg.get("cal_vatc_v2", False)
        # cal_vatc_v3 = cfg.get("cal_vatc_v3", False)
        # scst = cfg.get("scst", False)
        # vqa = cfg.get("vqa", False)
        # loss_weight_dict = cfg.get("loss_weight_dict", {})

        loss_config = cfg.get("loss_config", {
            'ITM': {'weight': 1.0},
            'ITC': {'weight': 1.0},
            'LM': {'weight': 1.0},
        })

        encoder_config = cfg.get("encoder_config", {'end2end': False, 'dim_features': 1024})
        visual_temporal_encoder_config = cfg.get("visual_temporal_encoder_config", None)
        visual_local_query_config = cfg.get("visual_local_query_config", None)
        query_temporal_embedding_config = cfg.get('query_temporal_embedding_config', None)
        max_txt_len = cfg.get("max_txt_len", 32)
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        vocab_size = cfg.get("vocab_size", 30523)
        token_drop_rate = cfg.get("token_drop_rate", 1.0)
        visual_num_temporal_embedding = cfg.get("visual_num_temporal_embedding", None)
        visual_num_spatial_embedding = cfg.get("visual_num_spatial_embedding", None)
        discard_visual_cls_token = cfg.get("discard_visual_cls_token", False)
        audio = cfg.get("audio", False)
        audio_dim_features = cfg.get("audio_dim_features", 768)
        audio_max_len = cfg.get("audio_max_len", 8)
        freeze_modules = cfg.get("freeze_modules", None)
        reset_pretrain_queries = cfg.get("reset_pretrain_queries", False)

        model = cls(
            # cal_itm=cal_itm,
            # cal_itc=cal_itc,
            # cal_lm=cal_lm,
            # cal_vac=cal_vac,
            # cal_vatc=cal_vatc,
            # cal_vatc_v2=cal_vatc_v2,
            # cal_vatc_v3=cal_vatc_v3,
            # scst=scst,
            # vqa=vqa,
            # loss_weight_dict=loss_weight_dict,
            loss_config=loss_config,
            encoder_config=encoder_config,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            vocab_size=vocab_size,
            max_txt_len=max_txt_len,
            visual_num_temporal_embedding=visual_num_temporal_embedding,
            visual_num_spatial_embedding=visual_num_spatial_embedding,
            discard_visual_cls_token=discard_visual_cls_token,
            audio=audio,
            audio_dim_features=audio_dim_features,
            audio_max_len=audio_max_len,
            visual_temporal_encoder_config=visual_temporal_encoder_config,
            freeze_modules=freeze_modules,
            visual_local_query_config=visual_local_query_config,
            query_temporal_embedding_config=query_temporal_embedding_config,
            reset_pretrain_queries=reset_pretrain_queries,
            token_drop_rate=token_drop_rate,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def forward(self, samples):
        """
        Args:
            samples: Dict
                text_input: processed caption
                image_id: image_id
                feature_xxx: feature
                    visual shape: B T K C
                    audio shape: B T C

        Returns:
            BlipOutput
        """
        if 'SCST' in self.loss_config:
            return self.scst_forward(samples)
        elif 'VQA' in self.loss_config:
            return self.vqa_forward(samples)

        video_embeds, video_atts, audio_embeds, audio_atts = self.prepare_video(samples)
        bs, device = video_embeds.shape[0], video_embeds.device
        num_query = self.num_query_token

        attention_mask = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(device)
        if audio_atts is not None:
            attention_mask = torch.cat([attention_mask, audio_atts], dim=1)

        query_tokens = self.query_tokens.expand(bs, -1, -1)  # B, num_query, d_model
        if self.query_temporal_embeddings is not None:
            query_tokens = self._add_te_to_query_tokens(query_tokens)

        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()
        video_mix_output = self.Qformer.bert(
            query_embeds=query_tokens,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,  # cross-attn
            encoder_attention_mask=video_atts,  # cross-attn
            use_cache=True,  # save past_key_values for LM
            return_dict=True,
        )
        # ender.record()
        # torch.cuda.synchronize()
        # time_ms = starter.elapsed_time(ender)
        # with open("encode time.txt", "a+") as f:
        #     f.write(f"{time_ms}\n")

        text_tokens = self.tokenizer(
            samples["text_input"],
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # Forward
        loss_itc, loss_itm, loss_lm, loss_vac, loss_dal = \
            [torch.zeros(1, device=device, dtype=torch.float) for _ in range(5)]
        public_parameters = {}
        if "LM" in self.loss_config:
            loss_lm = self.forward_language_modeling(
                video_mix_output, text_tokens,
                audio_embeds=audio_embeds, audio_atts=audio_atts,
                device=device
            )
            loss_lm *= self.loss_config['LM']['weight']
        if {"VATC", "AVTC", "att-VTC", "topk-VATC", "e-topk-VATC"} & set(self.loss_config.keys()):
            visual_output, audio_output, text_output = self.forward_video_sep_encode(
                query_tokens, audio_embeds, audio_atts,
                text_tokens.input_ids, text_tokens.attention_mask,
                video_embeds, video_atts
            )
            public_parameters['visual_output'] = visual_output
            public_parameters['audio_output'] = audio_output
            public_parameters['text_output'] = text_output

        if "ITC" in self.loss_config:
            # only text, for ITC
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            video_low_dim = F.normalize(
                self.vision_proj(video_mix_output.last_hidden_state[:, :num_query, :]), dim=-1
            )
            text_low_dim = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
            loss_itc, sim_t2i, sim_i2t = self.forward_image_text_contrastive(
                video_low_dim, text_low_dim,
            )
            public_parameters['sim_t2i'] = sim_t2i
            public_parameters['sim_i2t'] = sim_i2t
            loss_itc *= self.loss_config['ITC']['weight']
        if "VAC" in self.loss_config:
            loss_vac = self.forward_visual_audio_contrastive(
                query_tokens, audio_embeds, audio_atts, video_embeds, video_atts
            ) * self.loss_config['VAC']['weight']
        if "VATC" in self.loss_config:
            loss_a2v, loss_a2t, loss_vtc, sim_i2t, sim_t2i = self.forward_visual_audio_text_contrastive(
                public_parameters['visual_output'],
                public_parameters['audio_output'],
                public_parameters['text_output']
            )
            public_parameters['sim_t2i'] = sim_t2i
            public_parameters['sim_i2t'] = sim_i2t
            loss_itc = self.loss_config['VATC']['itc_weight'] * loss_vtc
            loss_vac = self.loss_config['VATC']['vac_weight'] * (loss_a2v + loss_a2t)
        if "AVTC" in self.loss_config:
            loss_avtc = self.forward_fine_grain_avt_contrastive(
                public_parameters['visual_output'],
                public_parameters['audio_output'],
                public_parameters['text_output'],
                audio_mask=audio_atts, text_mask=text_tokens.attention_mask
            )
            loss_vac = self.loss_config['AVTC']['weight'] * loss_avtc
        if "att-VTC" in self.loss_config:
            loss_vtc, sim_i2t, sim_t2i = self.forward_att_vtc(
                public_parameters['visual_output'],
                public_parameters['audio_output'],
                public_parameters['text_output']
            )
            public_parameters['sim_t2i'] = sim_t2i
            public_parameters['sim_i2t'] = sim_i2t
            loss_itc = self.loss_config['att-VTC']['weight'] * loss_vtc
        if "topk-VTC" in self.loss_config:
            loss_vtc, sim_i2t, sim_t2i = self.forward_topk_VTC(
                video_mix_output, text_tokens,
                top_k=self.loss_config['topk-VTC']['top_k'],
                top_p=self.loss_config['topk-VTC']['top_p'],
            )
            public_parameters['sim_t2i'] = sim_t2i
            public_parameters['sim_i2t'] = sim_i2t
            loss_itc = self.loss_config['topk-VTC']['weight'] * loss_vtc
        if "topk-VATC" in self.loss_config:
            loss_topk_avtc, sim_t2v, sim_v2t = self.forward_topk_vatc(
                public_parameters['visual_output'],
                public_parameters['audio_output'],
                public_parameters['text_output'],
            )
            loss_vac = loss_topk_avtc * self.loss_config['topk-VATC']['weight']
            public_parameters['sim_t2i'] = sim_t2v
            public_parameters['sim_i2t'] = sim_v2t
        if "e-topk-VATC" in self.loss_config:
            # text_output = self.Qformer.bert(
            #     text_tokens.input_ids,
            #     attention_mask=text_tokens.attention_mask,
            #     return_dict=True,
            # )
            # loss_topk_avtc, sim_t2v, sim_v2t = self.forward_efficient_topk_vatc(
            #     video_mix_output.last_hidden_state[:, :num_query, :],
            #     video_mix_output.last_hidden_state[:, num_query:, :],
            #     text_output.last_hidden_state,
            # )
            loss_topk_avtc, sim_t2v, sim_v2t = self.forward_efficient_topk_vatc(
                public_parameters['visual_output'],
                public_parameters['audio_output'],
                public_parameters['text_output'],
            )
            loss_vac = loss_topk_avtc * self.loss_config['e-topk-VATC']['weight']
            public_parameters['sim_t2i'] = sim_t2v
            public_parameters['sim_i2t'] = sim_v2t

        if "ITM" in self.loss_config:
            if self.audio:
                loss_itm = self.forward_video_text_matching(
                    video_embeds, text_tokens,
                    public_parameters['sim_t2i'], public_parameters['sim_i2t'],
                    audio_embeds, audio_atts
                )
            else:
                loss_itm = self.forward_image_text_matching(
                    video_embeds, text_tokens,
                    public_parameters['sim_t2i'], public_parameters['sim_i2t'],
                )
            loss_itm *= self.loss_config['ITM']['weight']

        if "DAL" in self.loss_config:
            if hasattr(self, 'visual_local_query_config') and self.visual_local_query_config is not None:
                groups = int(self.visual_local_query_config['num_frames'] * 16 * 16
                             / math.prod(self.visual_local_query_config['local_query_windows_size']))
            else:
                groups = 0
            loss_dal = self.forward_query_disagreement(
                video_mix_output.last_hidden_state[:, :self.num_query_token],
                groups=groups
            )
            loss_dal *= self.loss_config['DAL']['weight']

        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm + loss_vac + loss_dal,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_vac=loss_vac,
            loss_dal=loss_dal
        )

    def forward_image_text_contrastive(self, image_feats, text_feat, device=None, bs=None):
        device = image_feats.device if device is None else device
        bs = image_feats.size(0) if bs is None else bs
        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs).to(device, dtype=int)

        loss_itc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                   ) / 2
        return loss_itc, sim_t2i, sim_i2t

    def forward_image_text_matching(self, video_embeds, text_tokens, sim_t2i, sim_i2t,
                                    device=None, bs=None, rank=None):
        device = video_embeds.device if device is None else device
        bs = video_embeds.size(0) if bs is None else bs
        rank = dist.get_rank() if rank is None else rank
        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(video_embeds)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [video_embeds, image_embeds_neg, video_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        return loss_itm

    def forward_video_text_matching(self, video_embeds, text_tokens, sim_t2i, sim_i2t,
                                    audio_embeds, audio_atts,
                                    device=None, bs=None, rank=None):
        device = video_embeds.device if device is None else device
        bs = video_embeds.size(0) if bs is None else bs
        rank = dist.get_rank() if rank is None else rank
        # ##============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        video_embeds_world = all_gather_with_grad(video_embeds)
        audio_embeds_world = all_gather_with_grad(audio_embeds)
        audio_atts_world = all_gather_with_grad(audio_atts)
        with torch.no_grad():
            # if sim_t2i.isnan().sum() + sim_i2t.isnan().sum() > 0:
            #     print(sim_t2i, sim_i2t)
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(0)

        # select a hard negative image for each text
        video_embeds_neg = []
        audio_embeds_neg, audio_atts_neg = [], []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # TRICK! using multinomial to avoid no grad
            video_embeds_neg.append(video_embeds_world[neg_idx])
            audio_embeds_neg.append(audio_embeds_world[neg_idx])
            audio_atts_neg.append(audio_atts_world[neg_idx])
        video_embeds_neg = torch.stack(video_embeds_neg, dim=0)
        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)
        audio_atts_neg = torch.stack(audio_atts_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Construct inputs, total length = 3 * batch_size
        # The first batch_size is POS video + POS text
        # The second batch_size is NEG video + POS text
        # The third batch_size is POS video + NEG text
        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )
        video_embeds_all = torch.cat(
            [video_embeds, video_embeds_neg, video_embeds], dim=0
        )  # pos, neg, pos
        video_atts_all = torch.ones(video_embeds_all.size()[:-1], dtype=torch.long, device=device)
        audio_embeds_all = torch.cat(
            [audio_embeds, audio_embeds_neg, audio_embeds], dim=0
        )  # pos, neg, pos
        audio_atts_all = torch.cat(
            [audio_atts, audio_atts_neg, audio_atts], dim=0
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=device)
        attention_mask_all = torch.cat([query_atts_itm, audio_atts_all, text_atts_all], dim=1)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            audio_embeds=audio_embeds_all,
            attention_mask=attention_mask_all,
            encoder_hidden_states=video_embeds_all,
            encoder_attention_mask=video_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1) + audio_embeds_all.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        return loss_itm

    def forward_language_modeling(self, query_output, text_tokens,
                                  audio_embeds=None, audio_atts=None, device=None):
        bs = query_output.last_hidden_state.shape[0]
        device = self.device if device is None else device
        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(
            device
        )
        if audio_atts is not None:
            attention_mask = torch.cat([query_atts, audio_atts, text_tokens.attention_mask], dim=1)
        else:
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            audio_embeds=audio_embeds,
            # audio_atts=audio_atts,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return loss_lm

    def forward_video_sep_encode(self, query_tokens: torch.Tensor,
                                 audio_embeds: torch.Tensor, audio_atts: torch.Tensor,
                                 text_tokens: torch.Tensor, text_tokens_mask: torch.Tensor,
                                 video_embeds, video_atts):
        # prepare 3D attention mask
        # |q_len |a_len|t_len|
        # | 1 1 1 0 0 0 0 |
        # | 1 1 1 0 0 0 0 |
        # | 1 1 1 0 0 0 0 |
        # | 0 0 0 1 1 0 0 |
        # | 0 0 0 1 1 0 0 |
        # | 0 0 0 0 0 1 1 |
        # | 0 0 0 0 0 1 1 |
        bs, device = query_tokens.shape[0], query_tokens.device
        q_len, a_len, t_len = query_tokens.shape[1], audio_embeds.shape[1], text_tokens.shape[1]
        attention_mask = torch.zeros([bs, q_len + a_len + t_len, q_len + a_len + t_len], dtype=torch.int, device=device)
        attention_mask[:, :q_len, :q_len] = 1
        attention_mask[:, q_len:q_len + a_len, q_len:q_len + a_len] = repeat(audio_atts, 'b l -> b s l', s=a_len)
        attention_mask[:, q_len + a_len:, q_len + a_len:] = repeat(text_tokens_mask, 'b l -> b s l', s=t_len)

        # forward
        # video_sep_output = checkpoint(self.Qformer.bert,
        #                               text_tokens, attention_mask, None, None,
        #                               query_tokens, audio_embeds,
        #                               video_embeds, video_atts,
        #                               None, None, None, None, True)
        video_sep_output = self.Qformer.bert(
            text_tokens,
            query_embeds=query_tokens,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,  # cross-attn
            encoder_attention_mask=video_atts,  # cross-attn
            use_cache=False,
            return_dict=True,
        )
        return video_sep_output.last_hidden_state.split([q_len, a_len, t_len], dim=1)

    def vqa_forward(self, samples, use_prompt=True):
        video_embeds, video_atts, audio_embeds, audio_atts = self.prepare_video(samples)
        bs, device = video_embeds.shape[0], video_embeds.device

        query_tokens = self.query_tokens.expand(bs, -1, -1)  # B, num_query, d_model
        if self.query_temporal_embeddings is not None:
            query_tokens = self._add_te_to_query_tokens(query_tokens)

        if use_prompt:
            question_input = ["Question:" + i for i in samples["text_input"]]
            answer_input = ["Answer:" + i for i in samples["answers"]]
        else:
            question_input = samples["text_input"]
            answer_input = samples["answers"]

        question_tokens = self.tokenizer(
            question_input,
            padding="longest",
            truncation=True,
            max_length=50,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        question_tokens.input_ids = torch.cat([
            torch.ones(bs, 1, dtype=torch.long, device=device).fill_(self.tokenizer.bos_token_id),
            question_tokens.input_ids,
        ], dim=1)
        question_tokens.attention_mask = torch.cat([
            torch.ones(bs, 1, dtype=torch.long, device=device),
            question_tokens.attention_mask,
        ], dim=1)
        answer_tokens = self.tokenizer(
            answer_input,
            padding="longest",
            truncation=True,
            max_length=10,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        answer_tokens.input_ids = answer_tokens.input_ids[:, 1:]
        answer_tokens.attention_mask = answer_tokens.attention_mask[:, 1:]

        attention_mask = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(device)
        if audio_atts is not None:
            attention_mask = torch.cat([attention_mask, audio_atts], dim=1)
        attention_mask = torch.cat([attention_mask, question_tokens.attention_mask], dim=1)
        # attention_mask = attention_mask.unsqueeze(dim=2).expand(-1, -1, attention_mask.shape[1])

        video_mix_output = self.Qformer.bert(
            question_tokens.input_ids,
            query_embeds=query_tokens,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,  # cross-attn
            encoder_attention_mask=video_atts,  # cross-attn
            use_cache=True,  # save past_key_values for LM
            return_dict=True,
        )

        # input_ids = text_tokens.labels.clone()
        # input_ids[:, 0] = self.tokenizer.pad_token_id
        labels = answer_tokens.input_ids.clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        if use_prompt:
            labels[:, :2] = -100
        attention_mask = torch.cat([attention_mask, answer_tokens.attention_mask], dim=1)
        # attention_mask = torch.cat([
        #     attention_mask,
        #     (~torch.eq(labels, self.tokenizer.pad_token_id)).long()
        # ], dim=1)

        lm_output = self.Qformer(
            answer_tokens.input_ids,
            # audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            past_key_values=video_mix_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return BlipOutput(
            loss=loss_lm,
            loss_itc=None,
            loss_itm=None,
            loss_lm=loss_lm,
            loss_vac=None,
        )

    def forward_visual_audio_contrastive(self, query_tokens: torch.Tensor, audio_embeds: torch.Tensor,
                                         audio_atts: torch.Tensor, video_embeds, video_atts):
        """
        @param query_tokens: Tensor[B, query_len, dim_model]
        @param audio_embeds: Tensor[B, audio_max_len, dim_model]
        @param audio_atts: Tensor[B, audio_max_len]
        @param video_embeds: Tensor
        @param video_atts: Tensor
        @return:
        """
        device = query_tokens.device
        # prepare 3D attention mask
        # |q_len |a_len|
        # | 1 1 1 0 0 |
        # | 1 1 1 0 0 |
        # | 1 1 1 0 0 |
        # | 0 0 0 1 1 |
        # | 0 0 0 1 1 |
        bs = query_tokens.shape[0]
        q_len, a_len = query_tokens.shape[1], audio_embeds.shape[1]
        attention_mask = torch.zeros([bs, q_len + a_len, q_len + a_len], dtype=torch.int, device=device)
        attention_mask[:, :q_len, :q_len] = 1
        attention_mask[:, q_len:, q_len:] = repeat(audio_atts, 'b l -> b s l', s=a_len)

        # forward
        video_sep_output = self.Qformer.bert(
            query_embeds=query_tokens,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,  # cross-attn
            encoder_attention_mask=video_atts,  # cross-attn
            return_dict=True,
        )

        ###============== Visual-Audio Contrastive ===================###
        visual_low_dim = F.normalize(
            self.vision_proj(video_sep_output.last_hidden_state[:, :q_len, :]), dim=-1
        )  # B, q_len, embed_dim
        audio_low_dim = F.normalize(
            self.audio_low_dim_proj(video_sep_output.last_hidden_state[:, q_len:, :]), dim=-1
        )

        # [B, t, c] X [B, s, c] -> [B,t,s]
        sim_a2v = torch.einsum('btc,bsc->bts', audio_low_dim, visual_low_dim)
        _, max_idx = sim_a2v.max(-1)  # [B, t]
        sim_a2v /= self.vac_temp

        loss_vac = F.cross_entropy(sim_a2v.view(-1, q_len), max_idx.view(-1), label_smoothing=0.1)
        return loss_vac

    def forward_visual_audio_text_contrastive(self, visual_output, audio_output, text_output):
        q_len, t_len = visual_output.shape[1], text_output.shape[1]
        bs, device = visual_output.shape[0], visual_output.device

        visual_low_dim = F.normalize(self.vision_proj(visual_output), dim=-1)  # B, q_len, embed_dim
        audio_low_dim = F.normalize(self.audio_low_dim_proj(audio_output), dim=-1)  # B, a_len, embed_dim
        text_low_dim = F.normalize(self.text_proj(text_output), dim=-1)  # B, a_len, embed_dim

        # =========Audio to Visual=========
        # [B, t, c] X [B, s, c] -> [B,t,s]
        sim_a2v = torch.einsum('btc,bsc->bts', audio_low_dim, visual_low_dim)
        _, max_idx = sim_a2v.max(-1)  # [B, t]
        sim_a2v /= self.vac_temp
        loss_a2v = F.cross_entropy(sim_a2v.view(-1, q_len), max_idx.view(-1), label_smoothing=0.1)
        # =========Audio to Text=========
        sim_a2t = torch.einsum('btc,bsc->bts', audio_low_dim, text_low_dim)
        _, max_idx = sim_a2t.max(-1)  # [B, t]
        sim_a2t /= self.vac_temp
        loss_a2t = F.cross_entropy(sim_a2t.view(-1, t_len), max_idx.view(-1), label_smoothing=0.1)
        # =========Visual <--> Text=========
        visual_low_dim_all = concat_all_gather(visual_low_dim)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_cls_low_dim_all = concat_all_gather(text_low_dim[:, 0].contiguous())  # [batch_size*num_gpu, embed_dim]
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_q2t = torch.einsum('bqc,nc->bnq', visual_low_dim, text_cls_low_dim_all)
        sim_v2t, _ = sim_q2t.max(-1)  # [batch_size, batch_size*num_gpu]
        sim_v2t /= self.temp
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.einsum('bc,nqc->bnq', text_low_dim[:, 0], visual_low_dim_all)
        sim_t2v, _ = sim_t2q.max(-1)  # [batch_size, batch_size*num_gpu]
        sim_t2v /= self.temp
        # 0 1 2 3 | 4 5 6 7 | 8 9 10 11 for each rank
        rank: int = dist.get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs).to(device)
        loss_vtc = (
                           F.cross_entropy(sim_v2t, targets, label_smoothing=0.1) +
                           F.cross_entropy(sim_t2v, targets, label_smoothing=0.1)
                   ) / 2
        return loss_a2v, loss_a2t, loss_vtc, sim_v2t, sim_t2v

    def forward_fine_grain_avt_contrastive(self, visual_output, audio_output, text_output,
                                           visual_mask=None, audio_mask=None, text_mask=None):
        """
        Args:
            visual_output: B, q_len, D
            audio_output: B, a_len, D
            text_output: B, t_len, D
            visual_mask: B, q_len
            audio_mask: B, a_len
            text_mask: B, t_len
        Returns:

        """
        q_len, a_len, t_len = visual_output.shape[1], audio_output.shape[1], text_output.shape[1]
        bs, device = visual_output.shape[0], visual_output.device

        if visual_mask is None:
            visual_mask = visual_output.new_ones(bs, q_len)
        if audio_mask is None:
            audio_mask = audio_output.new_ones(bs, a_len)
        if text_mask is None:
            text_mask = text_output.new_ones(bs, t_len)

        visual_low_dim = F.normalize(self.vision_proj(visual_output), dim=-1)  # B, q_len, embed_dim
        audio_low_dim = F.normalize(self.audio_low_dim_proj(audio_output), dim=-1)  # B, a_len, embed_dim
        text_low_dim = F.normalize(self.text_proj(text_output), dim=-1)  # B, t_len, embed_dim

        # =========Audio to Visual=========
        mask_a2v = torch.logical_and(
            audio_mask.unsqueeze(2).repeat(1, 1, q_len),
            visual_mask.unsqueeze(1).repeat(1, a_len, 1),
        )
        # [B, t, c] X [B, s, c] -> [B,t,s]
        sim_a2v = torch.einsum('btc,bsc->bts', audio_low_dim, visual_low_dim) * mask_a2v
        _, max_idx = sim_a2v.max(-1)  # [B, a_len]
        max_idx.masked_fill_(~audio_mask, -100)
        sim_a2v = sim_a2v / self.vac_temp
        loss_a2v = F.cross_entropy(sim_a2v.view(-1, q_len), max_idx.view(-1), label_smoothing=0.1)
        # =========Audio to Text=========
        mask_a2t = torch.logical_and(
            audio_mask.unsqueeze(2).repeat(1, 1, t_len),
            text_mask.unsqueeze(1).repeat(1, a_len, 1),
        )
        sim_a2t = torch.einsum('btc,bsc->bts', audio_low_dim, text_low_dim) * mask_a2t
        _, max_idx = sim_a2t.max(-1)  # [B, a_len]
        max_idx.masked_fill_(~audio_mask, -100)
        sim_a2t = sim_a2t / self.vac_temp
        loss_a2t = F.cross_entropy(sim_a2t.view(-1, t_len), max_idx.view(-1), label_smoothing=0.1)
        # # =========Query to Text=========
        # mask_q2t = torch.logical_and(
        #     visual_mask.unsqueeze(2).repeat(1, 1, t_len),
        #     text_mask.unsqueeze(1).repeat(1, a_len, 1),
        # )
        # sim_q2t = torch.einsum('btc,bsc->bts', visual_low_dim, text_low_dim) * mask_q2t
        # _, max_idx = sim_q2t.max(-1)  # [B, a_len]
        # max_idx.masked_fill_(~visual_mask, -100)
        # sim_q2t = sim_q2t / self.vac_temp
        # loss_q2t = F.cross_entropy(sim_q2t.view(-1, t_len), max_idx.view(-1), label_smoothing=0.1)

        loss_avtc = (loss_a2t + loss_a2v) / 2
        # loss_avtc = (loss_a2t + loss_a2v + loss_q2t) / 3
        return loss_avtc

    def forward_topk_vatc(self, visual_output, audio_output, text_output):
        q_len, t_len = visual_output.shape[1], text_output.shape[1]
        bs, device = visual_output.shape[0], visual_output.device
        rank: int = dist.get_rank()
        # 0 1 2 3 | 4 5 6 7 | 8 9 10 11 for each rank
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs).to(device, dtype=int)
        ws: int = dist.get_world_size()
        top_k_query: int = self.loss_config['topk-VATC']['top_k_query']
        top_k_at: int = self.loss_config['topk-VATC']['top_k_at']
        top_k_qa: int = self.loss_config['topk-VATC']['top_k_qa']

        visual_low_dim = F.normalize(self.vision_proj(visual_output), dim=-1)  # B, q_len, embed_dim
        audio_low_dim = F.normalize(self.audio_low_dim_proj(audio_output), dim=-1)  # B, a_len, embed_dim
        text_low_dim = F.normalize(self.text_proj(text_output), dim=-1)  # B, a_len, embed_dim

        visual_low_dim_all = concat_all_gather(visual_low_dim)  # [batch_size*num_gpu, q_len, embed_dim]
        audio_low_dim_all = concat_all_gather(audio_low_dim)  # [batch_size*num_gpu, a_len, embed_dim]
        text_low_dim_all = concat_all_gather(text_low_dim)  # [batch_size*num_gpu, t_len, embed_dim]
        text_cls_low_dim_all = text_low_dim_all[:, 0]

        # =========Visual <--> Text=========
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_q2t = torch.einsum('bqc,nc->bnq', visual_low_dim, text_cls_low_dim_all)  # [b, bn, q_len]
        th_value = sim_q2t.topk(top_k_query, dim=-1).values[:, :, -1].unsqueeze(-1)  # [b, bn, 1]
        sim_q2t[sim_q2t < th_value] *= 1e-4
        sim_v2t = sim_q2t.mean(-1)
        # sim_v2t, _ = sim_q2t.max(-1)  # [batch_size, batch_size*num_gpu]
        sim_v2t /= self.vt_temp
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.einsum('bc,nqc->bnq', text_low_dim[:, 0], visual_low_dim_all)  # [b, bn, q_len]
        th_value = sim_t2q.topk(top_k_query, dim=-1).values[:, :, -1].unsqueeze(-1)  # [b, bn, 1]
        sim_t2q[sim_t2q < th_value] *= 1e-4
        sim_t2v = sim_t2q.mean(-1)
        # sim_t2v, _ = sim_t2q.max(-1)  # [batch_size, batch_size*num_gpu]
        sim_t2v /= self.vt_temp
        loss_vtc = (
                           F.cross_entropy(sim_v2t, targets, label_smoothing=0.1) +
                           F.cross_entropy(sim_t2v, targets, label_smoothing=0.1)
                   ) / 2

        # =========Visual <--> Audio=========
        sim_q2a = torch.einsum('bqc,nac->bnqa', visual_low_dim, audio_low_dim_all)  # [b, bn, q_len, a_len]
        sim_q2a = sim_q2a.reshape(bs, bs * ws, -1)  # [b, bn, q_len * a_len]
        th_value = sim_q2a.topk(top_k_qa, dim=-1).values[:, :, -1].unsqueeze(-1)  # [b, bn, 1]
        sim_q2a[sim_q2a < th_value] *= 1e-4
        sim_v2a = sim_q2a.mean(-1)
        sim_v2a /= self.va_temp
        sim_a2q = torch.einsum('nqc,bac->bnqa', visual_low_dim_all, audio_low_dim)  # [b, bn, q_len, a_len]
        sim_a2q = sim_a2q.reshape(bs, bs * ws, -1)  # [b, bn, q_len * a_len]
        th_value = sim_a2q.topk(top_k_qa, dim=-1).values[:, :, -1].unsqueeze(-1)  # [b, bn, 1]
        sim_a2q[sim_a2q < th_value] *= 1e-4
        sim_a2v = sim_a2q.mean(-1)
        sim_a2v /= self.va_temp
        loss_vac = (
                           F.cross_entropy(sim_v2a, targets, label_smoothing=0.1) +
                           F.cross_entropy(sim_a2v, targets, label_smoothing=0.1)
                   ) / 2
        # =========Audio <--> Text =========
        sim_t2a = torch.einsum('bqc,nac->bnqa', text_low_dim, audio_low_dim_all)  # [b, bn, t_len, a_len]
        sim_t2a = sim_t2a.reshape(bs, bs * ws, -1)  # [b, bn, q_len * a_len]
        th_value = sim_t2a.topk(top_k_at, dim=-1).values[:, :, -1].unsqueeze(-1)  # [b, bn, 1]
        sim_t2a[sim_t2a < th_value] *= 1e-4
        sim_t2a = sim_t2a.mean(-1)
        sim_t2a /= self.at_temp
        sim_a2t = torch.einsum('nqc,bac->bnqa', text_low_dim_all, audio_low_dim)  # [b, bn, t_len, a_len]
        sim_a2t = sim_a2t.reshape(bs, bs * ws, -1)  # [b, bn, q_len * a_len]
        th_value = sim_a2t.topk(top_k_at, dim=-1).values[:, :, -1].unsqueeze(-1)  # [b, bn, 1]
        sim_a2t[sim_a2t < th_value] *= 1e-4
        sim_a2t = sim_a2t.mean(-1)
        sim_a2t /= self.at_temp
        loss_atc = (
                           F.cross_entropy(sim_t2a, targets, label_smoothing=0.1) +
                           F.cross_entropy(sim_a2t, targets, label_smoothing=0.1)
                   ) / 2
        return (loss_vac + loss_vtc + loss_atc) / 3, sim_t2v, sim_v2t

    def forward_efficient_topk_vatc(self, visual_output, audio_output, text_output,
                                    visual_mask=None, audio_mask=None, text_mask=None):
        q_len, t_len = visual_output.shape[1], text_output.shape[1]
        bs, device = visual_output.shape[0], visual_output.device
        rank: int = dist.get_rank()
        top_k_query: int = self.loss_config['e-topk-VATC']['top_k_query']
        top_k_at: int = self.loss_config['e-topk-VATC']['top_k_at']
        top_k_qa: int = self.loss_config['e-topk-VATC']['top_k_qa']

        visual_low_dim = F.normalize(self.vision_proj(visual_output), dim=-1).half()  # B, q_len, embed_dim
        audio_low_dim = F.normalize(self.audio_low_dim_proj(audio_output), dim=-1).half()  # B, a_len, embed_dim
        text_low_dim = F.normalize(self.text_proj(text_output), dim=-1).half()  # B, t_len, embed_dim
        text_cls_low_dim = text_low_dim[:, 0, :].unsqueeze(1)

        # =========Visual <--> Text=========
        loss_vtc, sim_v2t, sim_t2v = top_k_batch_NCE_loss(visual_low_dim, text_cls_low_dim, top_k=top_k_query,
                                                          sim_temp=self.vt_temp)
        # =========Visual <--> Audio=========
        loss_vac = top_k_intra_NCE_loss(visual_low_dim, audio_low_dim, top_k=top_k_qa, sim_temp=self.va_temp)
        # =========Text   <--> Audio=========
        loss_atc = top_k_intra_NCE_loss(text_low_dim, audio_low_dim, top_k=top_k_at, sim_temp=self.at_temp)
        # if loss_vac <= 0.1 or loss_atc <= 0.1:
        # _debug_loss_vac = loss_vac.detach().cpu().item()
        # _debug_loss_atc = loss_atc.detach().cpu().item()
        # _debug_loss_vtc = loss_vtc.detach().cpu().item()
        # print(_debug_loss_vac, _debug_loss_atc, _debug_loss_vtc)

        return (loss_vac + loss_vtc + loss_atc) / 3, sim_t2v, sim_v2t

    def forward_att_vtc(self, visual_output, audio_output, text_output):
        q_len, t_len = visual_output.shape[1], text_output.shape[1]
        bs, device = visual_output.shape[0], visual_output.device

        video_low_dim = F.normalize(
            self.vision_proj(
                self.pooler(
                    torch.cat([visual_output, audio_output], dim=1),
                    self.pooler_query
                )
            ), dim=-1
        )  # B, 1, embed_dim
        text_low_dim = F.normalize(self.text_proj(text_output), dim=-1)  # B, txt_len, embed_dim

        # =========Video <--> Text=========
        visual_low_dim_all = concat_all_gather(video_low_dim.squeeze(1))  # [batch_size*num_gpu, embed_dim]
        text_cls_low_dim_all = concat_all_gather(text_low_dim[:, 0].contiguous())  # [batch_size*num_gpu, embed_dim]
        # [batch_size, batch_size*num_gpu]
        sim_v2t = torch.einsum('bc,nc->bn', video_low_dim.squeeze(1), text_cls_low_dim_all) / self.temp
        # [batch_size, batch_size*num_gpu]
        sim_t2v = torch.einsum('bc,nc->bn', text_low_dim[:, 0], visual_low_dim_all) / self.temp
        # 0 1 2 3 | 4 5 6 7 | 8 9 10 11 for each rank
        rank: int = dist.get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs).to(device, dtype=int)
        loss_vtc = (
                           F.cross_entropy(sim_v2t, targets, label_smoothing=0.1) +
                           F.cross_entropy(sim_t2v, targets, label_smoothing=0.1)
                   ) / 2
        return loss_vtc, sim_v2t, sim_t2v

    def forward_topk_VTC(self, video_mix_output, text_tokens, top_k=5, top_p=0.8, device=None, bs=None):
        # only text, for ITC
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        video_low_dim = F.normalize(self.vision_proj(video_mix_output.last_hidden_state), dim=-1)  # [B, q+a, E]
        text_low_dim = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)  # [B, E]
        video_low_dim_all = concat_all_gather(video_low_dim)  # [batch_size*num_gpu, q+a, embed_dim]
        text_feat_all = concat_all_gather(text_low_dim)  # [batch_size*num_gpu, embed_dim]

        device = video_low_dim.device if device is None else device
        bs: int = video_low_dim.size(0) if bs is None else bs
        # 0 1 2 3 | 4 5 6 7 | 8 9 10 11 for each rank
        rank: int = dist.get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs).to(device, dtype=int)

        def _sim_mat_top_k_p(sim: torch.Tensor, dim: int) -> torch.Tensor:
            if dim != len(sim.shape) - 1:
                sim = sim.transpose(dim, -1)
            with torch.no_grad():
                # [B,q,BN] -> [B,q] -> [B,top_k]
                # choose positive pair, calculate the probability, get top K prob queries
                tgt = targets[:, None, None].expand(-1, sim.shape[1], 1)
                prob, prob_idx = torch.softmax(sim.gather(-1, tgt).squeeze(-1), dim=1).topk(top_k, dim=1)
                # get top P queries
                prob_mask = torch.cumsum(prob, dim=-1) > top_p
                prob_mask[:, 0] = False  # keep at least one valid input
                prob.masked_fill_(prob_mask, 0.0)
                # normalize to avoid too small activation
                prob = F.normalize(prob, dim=-1)
                # add by weight
                weight = torch.zeros(sim.shape[:2], dtype=sim.dtype, device=sim.device)
                weight.scatter_(1, index=prob_idx, src=prob.to(dtype=sim.dtype))
            return (sim * weight.unsqueeze(-1)).mean(1)

        sim_v2t = torch.einsum('bqc,nc->bqn', video_low_dim.squeeze(1), text_feat_all) / self.temp
        if sim_v2t.isnan().sum() > 0:
            print(f"111 {self.temp}\n{sim_v2t}")
            print(f"{video_low_dim.isnan().sum()}, {text_feat_all.isnan().sum()}")
            raise ValueError
        # [b,q,n] -> [b,q] -> [b, top_k]
        sim_v2t = _sim_mat_top_k_p(sim_v2t, dim=2)
        if sim_v2t.isnan().sum() > 0:
            print(f"222 {self.temp}\n{sim_v2t}")
            print(f"{video_low_dim.isnan().sum()}, {text_feat_all.isnan().sum()}")
            raise ValueError

        sim_t2v = torch.einsum('bc,nqc->bnq', text_low_dim, video_low_dim_all) / self.temp
        if sim_t2v.isnan().sum() > 0:
            raise ValueError
        # [b,n,q] -> [b,q] -> [b, top_k]
        sim_t2v = _sim_mat_top_k_p(sim_t2v, dim=1)
        if sim_t2v.isnan().sum() > 0:
            raise ValueError
        loss_vtc = (
                           F.cross_entropy(sim_v2t, targets, label_smoothing=0.1) +
                           F.cross_entropy(sim_t2v, targets, label_smoothing=0.1)
                   ) / 2
        return loss_vtc, sim_t2v, sim_v2t

    def forward_query_disagreement(self, query_output: torch.Tensor, groups: int = None):
        """
        make cosine similarity < self.loss_config['DAL']['threshold']
        Args:
            groups: int
            query_output: Tensor[B, num_query, D]
        Returns:
            disagreement loss: Tensor[1]
        """
        # if has group: B,Q,d -> Bg,Q/g,d -> Bg,Q/g,Q/g
        if groups is not None:
            query_output = rearrange(query_output, 'b (g q) d -> (b g) q d', g=groups)

        # B,1,Q,d @ B,Q,d,1
        query_output = F.normalize(query_output, dim=-1)
        sim = torch.einsum('bqd,bpd->bqp', query_output, query_output)
        # sim.fill_diagonal_(1.0)
        # diagnal mask
        sim *= (1 - torch.eye(sim.shape[1], device=sim.device))
        loss_disagree = torch.clamp(sim.abs() - self.loss_config['DAL']['threshold'], min=0).mean()
        return loss_disagree

    def scst_forward(self, samples, max_length=30, min_length=5, num_beams=3):
        """
        Do SCST training forward:
            1. prepare inputs
            2. cache video outputs
            3. Do sample generation, get sample probability and captions
            4. Do beam search generation, get captions
            5. calculate CIDEr score
            6. return loss
        @param samples:
        @param max_length:
        @param min_length:
        @param num_beams:
        @return:
        """
        # 1. prepare inputs
        gts = samples["gts"]
        video_embeds, video_atts, audio_embeds, audio_atts = self.prepare_video(samples)

        # basic info about train samples
        device = video_embeds.device
        bs = video_embeds.shape[0]
        # Attention mask of query and audio (video)
        attention_mask = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(device)
        if audio_atts is not None:
            attention_mask = torch.cat([attention_mask, audio_atts], dim=1)

        # 2. cache video outputs
        # video_mix_output.past_key_values: 12 layer tuple
        # (key, value)
        # batch, n_head, 32query|40q+au, embed_per_head(64=768/12)
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.query_temporal_embeddings is not None:
            query_tokens = self._add_te_to_query_tokens(query_tokens)
        video_mix_output = self.Qformer.bert(
            query_embeds=query_tokens,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,  # cross-attn
            encoder_attention_mask=video_atts,  # cross-attn
            use_cache=True,  # save past_key_values for LM
            return_dict=True,
        )

        # prepare generation
        gen_start_attention_mask = torch.cat([
            attention_mask,
            torch.ones((bs, 1), dtype=torch.long, device=device)
        ], dim=-1)  # BOS
        model_kwargs = {
            "encoder_hidden_states": video_embeds,
            "encoder_attention_mask": video_atts,
            "attention_mask": gen_start_attention_mask,
            "past_key_values": video_mix_output.past_key_values
        }
        if self.audio:
            # model_kwargs['audio_embeds'] = audio_embeds
            model_kwargs['audio_atts'] = audio_atts
        bos_start_input_ids = (
            torch.LongTensor(bs, 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(device)
        )

        # 3. Do sample generation, get sample probability and captions
        sample_outputs = self.Qformer.generate_with_grad(
            input_ids=bos_start_input_ids,
            query_embeds=None,
            audio_embeds=None,
            max_length=max_length,
            min_length=min_length,
            num_beams=1,
            do_sample=True,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **model_kwargs
        )
        sample_captions = self.tokenizer.batch_decode(sample_outputs.sequences, skip_special_tokens=True)  # B, length
        sample_all_probs = torch.softmax(torch.stack(sample_outputs.scores, dim=1), dim=-1)  # B, length, vocab_size
        sample_probs = torch.gather(sample_all_probs, index=sample_outputs.sequences.unsqueeze(-1)[:, 1:],
                                    dim=2)  # B, L, 1
        sample_mask = (sample_outputs.sequences.unsqueeze(-1)[:, 1:] != 0).long()  # B, L, 1
        log_sample_probs = (sample_probs + 1e-08).log() * sample_mask

        # 4. Do beam search generation, get captions
        for k, v in model_kwargs.items():
            if 'past_key_values' in k:
                model_kwargs[k] = self._expand_past_key_values_for_beam(v, num_beams)
        # del model_kwargs['past_key_values']
        beam_outputs = self.Qformer.generate(
            input_ids=bos_start_input_ids,
            query_embeds=None,
            audio_embeds=None,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        beam_captions = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

        # 5. calculate CIDEr score
        tokenized_gts = eval_tools.PTBTokenizer.tokenize(gts)  # dict{0: [xxx, xxx], 1: [yyy, yyy]}
        tokenized_samples = eval_tools.PTBTokenizer.tokenize(sample_captions)  # dict{0: xxx, 1: yyy}
        tokenized_beams = eval_tools.PTBTokenizer.tokenize(beam_captions)  # dict{0: xxx, 1: yyy}
        reward = eval_tools.compute_ciders(tokenized_gts, tokenized_samples)[1]
        reward_baseline = eval_tools.compute_ciders(tokenized_gts, tokenized_beams)[1]

        # 6. return loss
        reward_weight = - torch.tensor(reward - reward_baseline, dtype=torch.float32, device=device)  # B
        loss = (
                reward_weight[:, None, None].expand(-1, sample_probs.shape[1], -1)
                * log_sample_probs
        ).mean()  # (expand[B] * [B, L, 1]).mean
        loss.requires_grad_(True)
        return {'loss': loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=3,
            max_length=30,
            min_length=10,
            top_p=0.9,
            top_k=50,
            temperature=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            output_attentions=False,
            measure_throughput=False,
    ):
        """
        Args:
            output_attentions (bool): Whether to output cross attentions, if True, only greedy search.
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if measure_throughput is True:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

        video_embeds, video_atts, audio_embeds, audio_atts = self.prepare_video(samples)

        device = video_embeds.device
        bs = video_embeds.shape[0]

        attention_mask = torch.ones((bs, self.num_query_token), dtype=torch.long, device=device)
        if self.audio:
            attention_mask = torch.cat([attention_mask, audio_atts], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((bs, 1), dtype=torch.long, device=device)],
                                   dim=-1)  # BOS

        if not use_nucleus_sampling:
            video_embeds = video_embeds.repeat_interleave(num_beams, dim=0)
            video_atts = video_atts.repeat_interleave(num_beams, dim=0)
            if self.audio:
                audio_embeds = audio_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1

        model_kwargs = {
            "encoder_hidden_states": video_embeds,
            "encoder_attention_mask": video_atts,
            "attention_mask": attention_mask
        }
        if self.audio:
            model_kwargs['audio_embeds'] = audio_embeds
            model_kwargs['audio_atts'] = audio_atts

        input_ids = (
            torch.LongTensor(bs, 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(device)
        )
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
        if self.query_temporal_embeddings is not None:
            query_tokens = self._add_te_to_query_tokens(query_tokens)

        if output_attentions:
            # instantiate logits processors
            logits_processor = LogitsProcessorList(
                [MinLengthLogitsProcessor(min_length, eos_token_id=self.tokenizer.sep_token_id)]
            )
            stopping_criteria = StoppingCriteriaList(
                [MaxLengthCriteria(max_length=max_length)]
            )
            outputs = self.Qformer.greedy_search_hack(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                output_attentions=True,
                return_dict_in_generate=True,
                query_embeds=query_tokens,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_kwargs
            )
            captions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            captions = self._remove_space_when_chinese(captions)
            return captions, outputs.cross_attentions[0], outputs.attentions
        else:
            outputs = self.Qformer.generate(
                input_ids=input_ids,
                query_embeds=query_tokens,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                do_sample=use_nucleus_sampling,
                top_p=top_p, top_k=top_k, temperature=temperature,
                length_penalty=length_penalty,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                # output_scores=True,
                # return_dict_in_generate=True,
                **model_kwargs
            )
            if measure_throughput is True:
                ender.record()
                torch.cuda.synchronize()
                time_ms = starter.elapsed_time(ender)
                # print(time_ms)
                return time_ms
            captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            captions = self._remove_space_when_chinese(captions)
            return captions

    def generate_for_gradcam(self, samples, max_length=30, min_length=10, **kwargs):
        # prepare features
        with self.maybe_autocast():
            samples['feature_visual'] = self._prepare_visual_feature(samples['video'])
            del samples['video']
            video_embeds, video_atts = self._prepare_video_embeds(samples['feature_visual'], samples['feature_visual_mask'])

            audio_embeds, audio_atts = None, None
            if self.audio:
                samples['feature_audio'], samples['feature_audio_mask'] = \
                    self._prepare_audio_feature(samples['audio'], samples['audio_mask'])
                del samples['audio'], samples['audio_mask']
                audio_embeds, audio_atts = self._prepare_audio_embeds(
                    samples['feature_audio'], samples['feature_audio_mask']
                )

        device = video_embeds.device

        # make attention_mask
        attention_mask = torch.ones((1, self.num_query_token), dtype=torch.long, device=device)
        if self.audio:
            attention_mask = torch.cat([attention_mask, audio_atts], dim=-1)
        # attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)

        query_tokens = self.query_tokens
        if self.query_temporal_embeddings is not None:
            query_tokens = self._add_te_to_query_tokens(query_tokens)

        video_mix_output = self.Qformer.bert(
            query_embeds=query_tokens,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,  # cross-attn
            encoder_attention_mask=video_atts,  # cross-attn
            use_cache=True,  # save past_key_values for LM
            return_dict=True,
        )

        # # make inputs
        # model_kwargs = {
        #     "encoder_hidden_states": video_embeds,
        #     "encoder_attention_mask": video_atts,
        #     "attention_mask": attention_mask
        # }
        # if self.audio:
        #     model_kwargs['audio_embeds'] = audio_embeds
        #     model_kwargs['audio_atts'] = audio_atts

        # input_ids = (
        #     torch.LongTensor(1, 1)
        #     .fill_(self.tokenizer.bos_token_id)
        #     .to(device)
        # )
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
        past_key_values = video_mix_output.past_key_values
        eos_token = self.tokenizer.sep_token_id
        output_logits, output_idx = [], [self.tokenizer.bos_token_id]
        while True:
            input_ids = torch.LongTensor(1, 1).fill_(output_idx[-1]).to(device)
            lm_output = self.Qformer(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            past_key_values = lm_output.past_key_values
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
            # select new token
            # logits: 1,1,30523
            curr_logit, curr_idx = lm_output.logits.squeeze().max(dim=-1)
            if curr_idx == eos_token:
                if len(output_idx) < min_length:
                    second_choice = lm_output.logits.squeeze().topk(2, dim=-1)
                    curr_logit, curr_idx = second_choice.values[-1], second_choice.indices[-1]
                else:
                    output_logits.append(lm_output.logits)
                    output_idx.append(curr_idx.cpu().item())
                    break
            output_logits.append(lm_output.logits)
            output_idx.append(curr_idx.cpu().item())

        # if not hasattr(self.Qformer, 'generate_with_grad'):
        #     self._set_grad_of_generate()
        # outputs = self.Qformer.generate_with_grad(
        #     input_ids=input_ids,
        #     query_embeds=query_tokens,
        #     max_length=max_length,
        #     min_length=min_length,
        #     top_p=top_p,
        #     eos_token_id=self.tokenizer.sep_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     repetition_penalty=repetition_penalty,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     **model_kwargs
        # )
        # captions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        outputs = {
            'logits': torch.stack(output_logits).squeeze(),
            'caption': self.tokenizer.batch_decode([output_idx], skip_special_tokens=True)[0],
            'sequence': torch.tensor(output_idx),
        }
        return outputs

    def predict_answers(self, samples, use_prompt=True, **kwargs):
        video_embeds, video_atts, audio_embeds, audio_atts = self.prepare_video(samples)
        bs, device = video_embeds.shape[0], video_embeds.device

        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
        if self.query_temporal_embeddings is not None:
            query_tokens = self._add_te_to_query_tokens(query_tokens)

        if use_prompt:
            question_input = ["Question:" + i + "Answer:" for i in samples["text_input"]]
        else:
            question_input = samples["text_input"]

        question_tokens = self.tokenizer(
            question_input,
            padding="longest",
            truncation=True,
            max_length=50,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        question_tokens.input_ids = torch.cat([
            torch.ones(bs, 1, dtype=torch.long, device=device).fill_(self.tokenizer.bos_token_id),
            question_tokens.input_ids,
        ], dim=1)
        question_tokens.attention_mask = torch.cat([
            torch.ones(bs, 1, dtype=torch.long, device=device),
            question_tokens.attention_mask,
        ], dim=1)
        # text_tokens = self.tokenizer(
        #     samples["text_input"],
        #     padding="longest",
        #     truncation=True,
        #     max_length=50,
        #     return_tensors="pt",
        # ).to(device)
        # input_ids = torch.cat([
        #     text_tokens.input_ids,
        #     torch.zeros(bs, 1, dtype=torch.long, device=device).fill_(self.tokenizer.bos_token_id)
        # ], dim=1)

        attention_mask = torch.ones((bs, self.num_query_token), dtype=torch.long, device=device)
        if self.audio:
            attention_mask = torch.cat([attention_mask, audio_atts], dim=-1)
        attention_mask = torch.cat([attention_mask, question_tokens.attention_mask], dim=1)
        # attention_mask = torch.cat([attention_mask, text_tokens.attention_mask, attention_mask.new_ones(bs, 1)], dim=1)
        # attention_mask = attention_mask.unsqueeze(dim=2).expand(-1, -1, attention_mask.shape[1])

        model_kwargs = {
            "encoder_hidden_states": video_embeds,
            "encoder_attention_mask": video_atts,
            "attention_mask": attention_mask
        }
        if self.audio:
            model_kwargs['audio_embeds'] = audio_embeds
            model_kwargs['audio_atts'] = audio_atts

        outputs = self.Qformer.generate(
            input_ids=question_tokens.input_ids,
            query_embeds=query_tokens,
            min_length=1,
            max_length=100,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        answers_ids = outputs[:, question_tokens.input_ids.shape[1]:]
        answers = self.tokenizer.batch_decode(answers_ids, skip_special_tokens=True)
        # print(answers)
        return answers

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test
        device = self.device
        """
        Requires data_loader.dataset.text, model.tokenizer,
         model.forward_text, text_proj
         model.forward_image, vision_proj
         model.compute_itm
        """
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation:"

        logging.info("Computing features for evaluation...")
        start_time = time.time()

        # =====================================================================
        #                      Calculate Text Embeddings
        # =====================================================================
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i + text_bs)]
            text_input = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(device)
            text_feat = self.Qformer.bert(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                return_dict=True,
            ).last_hidden_state[:, 0, :]
            text_embed = F.normalize(self.text_proj(text_feat))
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)

        # =====================================================================
        #                      Calculate Visual Embeddings
        # =====================================================================
        video_feats = []
        video_low_dim_querys = []
        for batch_idx, samples in enumerate(data_loader):
            video_embeds, video_atts, audio_embeds, audio_atts = self.prepare_video(samples)
            bs, device = video_embeds.shape[0], video_embeds.device
            num_query = self.num_query_token

            attention_mask = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(device)
            if audio_atts is not None:
                attention_mask = torch.cat([attention_mask, audio_atts], dim=1)

            query_tokens = self.query_tokens.expand(bs, -1, -1)  # B, num_query, d_model
            if self.query_temporal_embeddings is not None:
                query_tokens = self._add_te_to_query_tokens(query_tokens)

            video_mix_output = self.Qformer.bert(
                query_embeds=query_tokens,
                audio_embeds=audio_embeds,
                attention_mask=attention_mask,
                encoder_hidden_states=video_embeds,  # cross-attn
                encoder_attention_mask=video_atts,  # cross-attn
                use_cache=False,  # save past_key_values for LM
                return_dict=True,
            )
            if hasattr(self, "pooler"):
                video_low_dim = F.normalize(
                    self.vision_proj(
                        self.pooler(
                            video_mix_output.last_hidden_state[:, :num_query, :],
                            self.pooler_query
                        )
                    ), dim=-1
                )  # B, 1, embed_dim
            else:
                video_low_dim = F.normalize(
                    self.vision_proj(video_mix_output.last_hidden_state[:, :num_query, :]),
                    dim=-1
                )
                if audio_embeds is not None and hasattr(self, 'audio_low_dim_proj'):
                    audio_low_dim = F.normalize(
                        self.audio_low_dim_proj(
                            video_mix_output.last_hidden_state[:, num_query:num_query + audio_embeds.shape[1], :]
                        ), dim=-1
                    )
                    video_low_dim = torch.cat([video_low_dim, audio_low_dim], dim=1)

            if dist_utils.is_dist_avail_and_initialized() and dist_utils.get_world_size() > 1:
                batch_feats = []
                for i, ts in enumerate((video_embeds, video_atts, audio_embeds, audio_atts, video_low_dim)):
                    if ts is not None:
                        ts = concat_all_gather(ts)  # [b1+b2, *]
                        dims = list(ts.shape[1:])
                        # [b1+b2, *] -> [n_gpu, bs, *] -> [bs, n_gpu, *] -> [-1, *]
                        ts = ts.view([-1, bs] + dims).transpose(0, 1).reshape([-1] + dims)
                        if batch_idx == len(data_loader.loader) - 1:  # remove extra samples. Multi-GPU bugs.
                            real_bs = data_loader.loader.batch_size * dist_utils.get_world_size()
                            num_samples = len(data_loader.dataset.image) % real_bs
                            if num_samples != 0:
                                ts = ts[:num_samples]
                    if i < 4:
                        batch_feats.append(ts.cpu() if ts is not None else ts)
                    else:
                        video_low_dim_querys.append(ts)
                video_feats.append(batch_feats)
            else:
                video_feats.append([i.cpu() if i is not None else None
                                    for i in (video_embeds, video_atts, audio_embeds, audio_atts)])
                video_low_dim_querys.append(video_low_dim)

        # refactor video_feats to a dict
        video_feats_dict = defaultdict(list)
        for batch_tuple in video_feats:
            video_feats_dict['video_embeds'].append(batch_tuple[0])
            video_feats_dict['video_atts'].append(batch_tuple[1])
            video_feats_dict['audio_embeds'].append(batch_tuple[2])
            video_feats_dict['audio_atts'].append(batch_tuple[3])
        video_feats = {}
        for k, v in video_feats_dict.items():
            if v[0] is None:
                video_feats[k] = None
            else:
                video_feats[k] = torch.cat(v, dim=0)
        video_low_dim_querys = torch.cat(video_low_dim_querys, dim=0)

        # =====================================================================
        #                    Compute Similarity Matrix
        # =====================================================================
        sims_matrix = []  # [all_video, all_text]
        for v_idx, video_low_dim in enumerate(video_low_dim_querys):
            # [q, E] @ [1000, E] -> [q, 1000]
            sim_q2t = video_low_dim @ text_embeds.t()
            if "topk-VTC" in self.loss_config:
                # [q, 1000] -> [q] -> [top_k]
                # choose the positive pair, calculate the probability, get top K prob queries
                prob, prob_idx = (torch.softmax(sim_q2t[:, v_idx], dim=0)
                                  .topk(self.loss_config['topk-VTC']['top_k'], dim=0))
                # get top P queries
                prob_mask = torch.cumsum(prob, dim=-1) > self.loss_config['topk-VTC']['top_p']
                prob_mask[0] = False  # keep at least one valid input
                prob.masked_fill_(prob_mask, 0.0)
                # normalize to avoid too small activation
                prob = F.normalize(prob, dim=-1)
                # add by weight
                weight = torch.zeros(sim_q2t.shape[0], dtype=sim_q2t.dtype, device=sim_q2t.device)
                weight.scatter_(0, index=prob_idx, src=prob.to(dtype=sim_q2t.dtype))
                sim_i2t = (sim_q2t * weight.unsqueeze(-1)).mean(0)
            elif "topk-VATC" in self.loss_config:
                top_k_query = self.loss_config['topk-VATC']['top_k_query']
                # th_value = sim_q2t.topk(top_k_query, dim=0).values[:, -1].unsqueeze(-1)  # [b, bn, 1]
                # sim_q2t[sim_q2t < th_value] *= 1e-4
                sim_i2t = sim_q2t.topk(top_k_query, dim=0).values.mean(0)
            else:
                sim_i2t, _ = sim_q2t.max(0)
            sims_matrix.append(sim_i2t)
        sims_matrix = torch.stack(sims_matrix, dim=0)

        # =====================================================================
        #                         Compute Matching(ITM)
        # =====================================================================
        num_tasks = dist_utils.get_world_size()
        rank = dist_utils.get_rank()

        # Video-to-Text
        # If 1000 for 2 gpus, step = 501, start = 0/501, end = 501/1000
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)
        score_matrix_i2t = torch.full(
            (len(data_loader.dataset.image), len(texts)), -100.0
        ).to(device)
        # For every single video, select top-k text
        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            video_embeds = video_feats['video_embeds'][start + i].repeat(k_test, 1, 1).to(device)
            video_atts = video_feats['video_atts'][start + i].repeat(k_test, 1).to(device)
            audio_embeds, audio_atts = None, None
            if video_feats['audio_embeds'] is not None:
                audio_embeds = video_feats['audio_embeds'][start + i].repeat(k_test, 1, 1).to(device)
                audio_atts = video_feats['audio_atts'][start + i].repeat(k_test, 1).to(device)
            score = self.compute_itm(
                video_embeds=video_embeds,
                audio_embeds=audio_embeds,
                audio_atts=audio_atts,
                text_ids=text_ids[topk_idx],
                text_atts=text_atts[topk_idx],
            ).float()
            # Finally sim = ITC_sim + ITM_sim
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

        # Text-to-Video
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full(
            (len(texts), len(data_loader.dataset.image)), -100.0
        ).to(device)
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)
        # For every single text, select top-k video
        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            video_embeds = video_feats['video_embeds'][topk_idx.cpu()].to(device)
            video_atts = video_feats['video_atts'][topk_idx.cpu()].to(device)
            audio_embeds, audio_atts = None, None
            if video_feats['audio_embeds'] is not None:
                audio_embeds = video_feats['audio_embeds'][topk_idx.cpu()].to(device)
                audio_atts = video_feats['audio_atts'][topk_idx.cpu()].to(device)
            score = self.compute_itm(
                video_embeds=video_embeds,
                audio_embeds=audio_embeds,
                audio_atts=audio_atts,
                text_ids=text_ids[start + i].repeat(k_test, 1),
                text_atts=text_atts[start + i].repeat(k_test, 1),
            ).float()
            score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()
            torch.distributed.all_reduce(
                score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    def compute_itm(self,
                    video_embeds, text_ids, audio_embeds=None,
                    audio_atts=None, text_atts=None):
        device, bs = video_embeds.device, video_embeds.shape[0]
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long, device=device)

        query_tokens = self.query_tokens.expand(bs, -1, -1)  # B, num_query, d_model
        if self.query_temporal_embeddings is not None:
            query_tokens = self._add_te_to_query_tokens(query_tokens)
        attention_mask = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long, device=device)
        if audio_atts is not None:
            attention_mask = torch.cat([attention_mask, audio_atts], dim=1)
        attention_mask = torch.cat([attention_mask, text_atts], dim=1)

        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        itm_len = query_tokens.size(1)
        if audio_embeds is not None:
            itm_len += audio_embeds.size(1)
        vl_embeddings = output_itm.last_hidden_state[:, : itm_len, :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    def prepare_video(self, samples):
        if self.end2end is True:
            with self.maybe_autocast():
                feature_visual = self._prepare_visual_feature(samples['video'])
                if type(feature_visual) is tuple:
                    samples['feature_visual'], samples['visual_sp_query_feature'] = feature_visual
                else:
                    samples['feature_visual'] = feature_visual
                del samples['video']
                if self.audio:
                    feature_audio = self._prepare_audio_feature(samples['audio'], samples['audio_mask'])
                    samples['feature_audio'], samples['feature_audio_mask'] = feature_audio
                    del samples['audio'], samples['audio_mask']

        patch_emb_idx = samples.get('feature_visual_mask', None)
        video_embeds, video_atts = self._prepare_video_embeds(samples['feature_visual'], samples['feature_visual_mask'],
                                                              patch_emb_idx=patch_emb_idx)
        audio_embeds, audio_atts = None, None
        if self.audio:
            audio_embeds, audio_atts = self._prepare_audio_embeds(samples['feature_audio'],
                                                                  samples['feature_audio_mask'])
        return video_embeds, video_atts, audio_embeds, audio_atts

    def _prepare_video_embeds(self, visual_feature: torch.Tensor, feature_visual_mask: torch.Tensor,
                              visual_sp_query_feature=None, patch_emb_idx: torch.Tensor = None):
        visual_feature = visual_feature.to(dtype=torch.float32)
        # if self.discard_visual_cls_token is True:
        #     visual_feature = visual_feature[:, :, 1:, :]

        # batch size, temporal, patch, feature dim
        # b, t, k, c = visual_feature.shape

        # batch size, frame length, feature dim
        b, k, c = visual_feature.shape

        # inject temporal information
        if self.visual_num_temporal_embedding is not None:
            self.visual_temporal_embeddings = self.pos_encoding(visual_feature)
            assert k <= self.visual_num_temporal_embedding, "input frames too much"
            # temp_emb = torch.stack([self.visual_temporal_embeddings[i] for i in range(t)], dim=0)  # t, c
            # visual_feature += repeat(temp_emb, 't c -> b t k c', b=b, k=k)
            temp_emb = torch.stack([self.visual_temporal_embeddings[i] for i in range(k)], dim=0)  # t, c
            visual_feature += repeat(temp_emb, 'k c -> b k c', b=b)

        video_embeds = self.ln_vision(visual_feature)
        if self.visual_temporal_encoder_config is not None:
            video_embeds = self.visual_temporal_encoder(video_embeds)

        # video_embeds = video_embeds.view([b, t * k, self.dim_features])
        # video_embeds = rearrange(video_embeds, 'b t k c -> b (t k) c')
        if visual_sp_query_feature is not None:
            visual_sp_query_feature = self.ln_vision(visual_sp_query_feature)
            video_embeds = torch.cat([video_embeds, visual_sp_query_feature], dim=1)
        video_atts = feature_visual_mask.to(dtype=torch.long)
        return video_embeds, video_atts

    def _prepare_audio_embeds(self, audio_feature, audio_atts):
        # batch size, temporal, feature dim
        b, t, c = audio_feature.shape

        # inject temporal information
        # assert t <= self.audio_max_len, "input audio frames too much"
        # temp_emb = torch.stack([self.audio_temporal_embeddings[i] for i in range(t)], dim=0)  # t, c
        # audio_feature += repeat(temp_emb, 't c -> b t c', b=b)

        audio_embeds = self.audio_proj(self.ln_audio(audio_feature))

        return audio_embeds, audio_atts

    def _prepare_visual_feature(self, video, chunk_split=1):
        """
        Args:
            video (Tensor): [B, T, C, H, W]
        Return:
            visual_feature (Tensor): [B, T, K, C]
        """
        bs, t = video.shape[0], video.shape[2]
        dtype = torch.float16 if self.encoder_config['vit_precision'] == 'fp16' else torch.float32
        assert (bs * t) % chunk_split == 0, f"Please make (batch_size * temporal) % {chunk_split} == 0"
        video = rearrange(video, 'b c t h w -> (b t) c h w').to(dtype=dtype)

        with torch.autocast(device_type='cpu' if str(video.device) == 'cpu' else 'cuda'):
            # visual_feature = apply_chunking_to_forward(
            #     lambda x: self.visual_encoder(x),
            #     bs * t // chunk_split,
            #     0,
            #     video,
            # )
            visual_feature = self.visual_encoder(video)
            if type(visual_feature) is tuple:
                visual_feature, visual_sp_query_feature = visual_feature  # (b t) p c, b q c
                visual_feature = rearrange(visual_feature, '(b t) p c -> b t p c', b=bs, t=t)
                return visual_feature, visual_sp_query_feature
            else:
                visual_feature = rearrange(visual_feature, '(b t) p c -> b t p c', b=bs, t=t)
                return visual_feature

    def _prepare_audio_feature(self, audio, audio_mask):
        # audio: B, T, c1, c2
        bs, t = audio.shape[0], audio.shape[1]
        # assert (bs * t) % 4 == 0, f"Please make (batch_size * temporal) % 4 == 0"
        audio = rearrange(audio, 'b t c1 c2 -> (b t) c1 c2')

        audio_feature = apply_chunking_to_forward(
            lambda x: self.audio_encoder(input_values=x).pooler_output,  # B, 768
            bs * t // 4,
            0,
            audio
        )
        # audio_feature = self.audio_encoder(input_values=audio).pooler_output
        audio_feature = rearrange(audio_feature, '(b t) c -> b t c', b=bs, t=t)
        return audio_feature, audio_mask

    def _expand_past_key_values_for_beam(self, past_key_values, expand_size):
        expanded = []
        for layer in past_key_values:
            k = layer[0].unsqueeze(dim=1).expand(-1, expand_size, -1, -1, -1).contiguous()
            k = rearrange(k, 'b e h p q -> (b e) h p q')
            v = layer[1].unsqueeze(dim=1).expand(-1, expand_size, -1, -1, -1).contiguous()
            v = rearrange(v, 'b e h p q -> (b e) h p q')
            expanded.append((k, v))
        return tuple(expanded)

    def _set_grad_of_generate(self):
        from types import MethodType
        undecorated = self.Qformer.generate.__closure__[0].cell_contents
        self.Qformer.generate_with_grad = MethodType(undecorated, self.Qformer)

    def _add_te_to_query_tokens(self, query_tokens: torch.Tensor):
        """
        self.query_temporal_embeddings: M, C
        @param query_tokens: B, K, C
        @return:
        """
        bs, num_query, _ = query_tokens.shape
        assert num_query % self.query_temporal_embeddings.size(0) == 0
        num_query_per_clip = num_query // self.query_temporal_embeddings.size(0)
        query_te = torch.repeat_interleave(self.query_temporal_embeddings, num_query_per_clip, dim=0)
        return query_tokens + query_te.unsqueeze(0).expand(bs, -1, -1)

    @staticmethod
    def _remove_space_when_chinese(captions):
        pattern = re.compile(u'[\u4e00-\u9fa5]')
        if pattern.search(captions[0]) is None:
            return captions
        else:
            return [i.replace(' ', '') for i in captions]
