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


