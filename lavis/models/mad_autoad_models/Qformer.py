import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor, device
from typing import Optional, Tuple, Dict, Any
import os
from einops import reduce, rearrange
from dataclasses import dataclass
import math

from ..blip2_models.Qformer import (
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertOnlyMLMHead
)
from lavis.models.val_qformer_models.generation_utils import GenerationMixinHack
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers import (
    BertPreTrainedModel,
    BertTokenizer
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class BaseModelOutputWithPastAndCrossAttentionsAndDisagreements(BaseModelOutputWithPastAndCrossAttentions):
    disagreements: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndDisagreements(BaseModelOutputWithPoolingAndCrossAttentions):
    disagreements: Optional[Tuple[torch.FloatTensor]] = None


class DisagreementRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        @param x: Tensor of Shape[B,num_query,d_model]
        @return: disagree: Tensor of Shape[B]
        """
        x = F.normalize(x, dim=-1)
        # B,1,Q,d @ B,Q,d,1
        sim = torch.einsum('bqd,bpd->bqp', x, x)
        # sum
        disagree = - reduce(sim, 'b q p -> b', 'mean')
        return disagree


class QformerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class QformerEmbeddings(nn.Module):
    """
    Construct the embeddings from word and position embeddings.
    Construct the embeddings of the inputs of self-attention.
    [query] [  audio  ] [    text      ]
                        [word_embedding]
            [   pos   ] [     pos      ]
    [        LayerNorm + Dropout       ]
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        if config.audio is True:
            self.audio_position_embeddings = nn.Embedding(
                config.audio_max_len, config.hidden_size
            )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.config = config

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            query_embeds=None,
            audio_embeds=None,
            pos_start_index=0,
    ):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                           :, pos_start_index: seq_length + pos_start_index
                           ].clone()

        all_embeds = []
        if query_embeds is not None:
            all_embeds.append(query_embeds)

        if audio_embeds is not None:
            audio_pid = self.position_ids[:, :audio_embeds.shape[1]].clone()
            audio_embeds = audio_embeds + self.audio_position_embeddings(audio_pid)
            all_embeds.append(audio_embeds)

        if input_ids is not None:
            word_embeds = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                word_embeds = word_embeds + position_embeddings
            all_embeds.append(word_embeds)

        embeddings = torch.cat(all_embeds, dim=1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def window_partition(x, ws: tuple):
    """
    Args:
        x: (B, T, H, W, C)
        ws: (4 ,4 ,4)
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // ws[0], ws[0],
        H // ws[1], ws[1],
        W // ws[2], ws[2],
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, math.prod(ws), C)
    return windows


class QformerLayerAudio(nn.Module):
    """
    Crucial Module of Qformer
    """

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num

        self.local_query_windows_size: tuple = getattr(self.config, 'local_query_windows_size', None)
        self.num_local_query: int = getattr(self.config, 'num_local_query', 0)
        self.num_frames: int = getattr(self.config, 'num_frames', 0)

        if self.config.add_cross_attention and (layer_num % self.config.cross_attention_freq == 0):
            self.crossattention = BertAttention(config, is_cross_attention=self.config.add_cross_attention)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

        if config.audio:
            # audio FeedForward
            self.intermediate_audio = BertIntermediate(config)
            self.output_audio = BertOutput(config)

        if config.disagree_regularize_config is not None:
            # if config.disagree_regularize_config.type == 'out':
            self.disagree = DisagreementRegularizer()
            self.disagree_type = config.disagree_regularize_config.type

    def forward_query_cross_attn(
            self, query_after_sa: torch.Tensor,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        """
        @param head_mask:
        @param encoder_hidden_states: b t*(hw+1) c
        @param encoder_attention_mask: b t*(hw+1)
        @param output_attentions:
        @param query_after_sa: Tensor[b (global+local) c]
        @return:
        """
        kwargs = dict(head_mask=head_mask, output_attentions=output_attentions)
        if self.local_query_windows_size is None:
            return self.crossattention(query_after_sa, encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_attention_mask, **kwargs)
        else:
            bs = query_after_sa.shape[0]
            query_length = query_after_sa.shape[1]
            global_query, local_query = torch.split(
                query_after_sa,
                [query_length - self.num_local_query, self.num_local_query], dim=1
            )  # b g c | b l c
            if global_query.shape[1] > 0:
                # 32,32,768 |optional attn 32,12,32,2056| (32,12,2056,64),(32,12,2056,64)
                global_output = self.crossattention(global_query, encoder_hidden_states=encoder_hidden_states,
                                                    encoder_attention_mask=encoder_attention_mask, **kwargs)
            else:
                global_output = (query_after_sa.new_empty(bs, 0, query_after_sa.shape[-1]), None)

            window_hid_s = rearrange(encoder_hidden_states, 'b (t k) c -> b t k c', t=self.num_frames)
            window_hid_s = window_hid_s[:, :, 1:, :]
            window_hid_s = rearrange(window_hid_s, 'b t (h w) c -> b t h w c', h=int(math.sqrt(window_hid_s.shape[2])))
            window_hid_s = window_partition(window_hid_s, ws=self.local_query_windows_size)  # (b num_win) patches c
            # Example:
            # num_local_query: 32
            # 8*16*16 patch -> 4*4*4 win_size * 32 num_win -> 1query/win
            # num_local_query: 64
            # 8*16*16 patch -> 4*4*4 win_size * 32 num_win -> 2query/win
            assert self.num_local_query % (window_hid_s.shape[0] // bs) == 0
            num_query_per_win = self.num_local_query // (window_hid_s.shape[0] // bs)
            # b (num_win * num_query_per_win) c -> (b num_win) num_query_per_win c
            local_query = rearrange(local_query, 'b (w q) c -> (b w) q c', q=num_query_per_win)
            local_output = self.crossattention(local_query, encoder_hidden_states=window_hid_s, **kwargs)

            # concat
            # 1024,1,768 | 1024,12,1,64 | 1024,12,64,64
            query_after_ca = torch.cat([
                global_output[0],
                rearrange(local_output[0], '(b w) q c -> b (w q) c', b=bs)
            ], dim=1)  # B, 64, 768
            # if kwargs['output_attentions'] is True:
            #     local_attn = rearrange(local_output[1], '(b w) head q ws -> b head q w ws', b=bs)
            #     n_head, n_win, size_win = local_attn.shape[1], local_attn.shape[3], local_attn.shape[4]
            #     factory = dict(device=local_attn.device, dtype=local_attn.dtype)
            #     all_query_attn = []
            #     for i in range(local_attn.shape[2]):
            #         attn = torch.zeros((bs, n_head, n_win, n_win, size_win), **factory)  # 32,12,32,32,64
            #         for j in range(n_win):
            #             attn[:, :, j, j, :] = local_attn[:, :, i, j, :]
            #         all_query_attn.append(torch.cat([torch.zeros(attn.shape[:4] + (1, ), **factory), attn], dim=-1))
            #     all_query_attn = torch.stack(all_query_attn, dim=3)  # 32,12,32,num_query_per_window,32,64
            #     # 32,12,32,2056
            #     all_query_attn = all_query_attn.view(bs, n_head, all_query_attn.shape[2] * all_query_attn.shape[3], -1)
            #     pass
            #     return query_after_ca, all_query_attn
            if kwargs['output_attentions'] is True:
                return (query_after_ca, (global_output[1], local_output[1]), None)
            return (query_after_ca, None)


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            query_length=0,
            audio_length=0,
    ):
        # 1. pass all hidden_states to Self-Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        # 2. pass different modality to Cross-Attention or FeedForward
        text_length = attention_output.shape[1] - query_length - audio_length
        query_attention_output, audio_attention_output, text_attention_output = torch.split(
            attention_output,
            [query_length, audio_length, text_length],
            dim=1
        )
        layer_output = []
        if query_length > 0:
            if self.has_cross_attention:
                assert (
                        encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers"
                # cross_attention_outputs = self.crossattention(
                #     query_attention_output,
                #     attention_mask,
                #     head_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                #     output_attentions=output_attentions,
                # )
                cross_attention_outputs = self.forward_query_cross_attn(
                    query_attention_output,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]

                # ADD disagreement regularization
                if self.config.disagree_regularize_config is not None and self.disagree_type == 'out':
                    disagreement = self.disagree(query_attention_output)  # [B]

                outputs = (
                        outputs + cross_attention_outputs[1:-1]
                )  # add cross attentions if we output attention weights
            query_layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )
            layer_output.append(query_layer_output)
        if audio_length > 0:
            audio_layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_audio,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                audio_attention_output,
            )
            layer_output.append(audio_layer_output)
        if text_length > 0:
            text_layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                text_attention_output,
            )
            layer_output.append(text_layer_output)
        layer_output = torch.cat(layer_output, dim=1)

        # 3. make output
        outputs = (layer_output,) + outputs
        if self.config.disagree_regularize_config is not None:
            outputs = outputs + (disagreement,)
        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_audio(self, attention_output):
        intermediate_output = self.intermediate_audio(attention_output)
        layer_output = self.output_audio(intermediate_output, attention_output)
        return layer_output


class QformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
            QformerLayerAudio(config, i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            query_length=0,
            audio_length=0
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_disagreements = (
            () if self.config.disagree_regularize_config is not None else None
        )

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, past_key_value, output_attentions, query_length
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                    audio_length
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            if self.config.disagree_regularize_config is not None:
                all_disagreements = all_disagreements + (layer_outputs[3],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                    all_disagreements
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentionsAndDisagreements(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            disagreements=all_disagreements,
        )


class QformerModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = QformerEmbeddings(config)

        self.encoder = QformerEncoder(config)

        self.pooler = QformerPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
            self,
            attention_mask: Tensor,
            input_shape: Tuple[int],
            device: device,
            is_decoder: bool,
            include_prev_tokens_mask: bool = False,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
                [B, all_token_len]
                all_token_len:
                    1. query text
                    2. query
                    3. text
                    4. query audio text
            is_decoder (bool):
                If is not decoder, no causal mask is made
            include_prev_tokens_mask (bool):
                Means `Does the output mask have query attention`
                When using past_key_value and generating texts, needs a [text_len, all_len] mask
                instead of a [all_len, text_len] mask
            input_shape (:obj:`Tuple[int]`):
                [B, text_len] the length of causal mask
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            """
            attention_mask: (Batch, 33) 33 = 32 + [CLS]
            """
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, text_length = input_shape
                all_token_length = attention_mask.shape[1]

                # make a simple causal mask. [B l l]
                seq_ids = torch.arange(text_length, device=device)
                causal_mask = (
                        seq_ids[None, None, :].repeat(batch_size, text_length, 1) <= seq_ids[None, :, None]
                ).to(attention_mask.dtype)

                # add a prefix ones mask to the causal mask
                if text_length < all_token_length:
                    prefix_length = all_token_length - text_length
                    if include_prev_tokens_mask:  # UniLM style attention mask
                        # [B,text_len,text_len] --> [B,all_len,text_len]
                        # add query attention
                        prefix_all_one_mask = torch.zeros(
                            (batch_size, prefix_length, text_length),
                            device=device,
                            dtype=causal_mask.dtype,
                        )
                        causal_mask = torch.cat([prefix_all_one_mask, causal_mask], dim=1)
                    # [..,..,text_len] -> [..,..,all_len]
                    prefix_all_one_mask = torch.ones(
                        (batch_size, causal_mask.shape[1], prefix_length),
                        device=device,
                        dtype=causal_mask.dtype,
                    )
                    causal_mask = torch.cat([prefix_all_one_mask, causal_mask], dim=-1)
                # multiply to merge mask
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            query_embeds=None,
            audio_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_decoder=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        # check multi-modality input
        if input_ids is None:
            assert (query_embeds is not None), "You have to specify query_embeds when input_ids is None"
        if past_key_values is not None:
            assert query_embeds is None and audio_embeds is None, "past_key_values dont need previous input"

        # past_key_values_length
        # past_key_values:
        # n_layers[B, heads, seq_len-1, dim_embed]
        # The length of past_key_values CONTAINS tokens of other modalities AND previous generated text tokens
        # SHIT, always 0
        # SHIT, not always 0
        audio_max_len = self.config.audio_max_len if self.config.audio is True else 0
        text_past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length - audio_max_len
            if past_key_values is not None
            else 0
        )
        query_length = query_embeds.shape[1] if query_embeds is not None else 0
        audio_length = audio_embeds.shape[1] if audio_embeds is not None else 0

        # embedding_output contains query,audio and part of the pre-generated text
        # May only contain 1 new input text token
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            audio_embeds=audio_embeds,
            pos_start_index=text_past_key_values_length,
        )

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        # *****************************************************************************
        # ************************ prepare self-attention mask ************************
        # Before going into self.get_extended_attention_mask() The attention_mask should
        # be [B, all_len], indicating which token is valid.
        # However, if attention_mask is missing in some case (during .generation()), a full
        # one mask will be created(not recommend)
        # *****************************************************************************
        # if attention_mask is None:
        #     attention_mask = torch.ones((batch_size, query_length), device=device)
        # if audio_atts is not None:
        #     attention_mask = torch.cat([attention_mask, audio_atts], dim=-1)
        # if attention_mask is None and text_past_key_values_length > 0:
        #     attention_mask = torch.cat([
        #         attention_mask,
        #         torch.ones([batch_size, text_past_key_values_length], device=device)
        #     ], dim=-1)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + text_past_key_values_length), device=device)
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # attention_mask contains the query,audio mask
        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_ids.shape,
                device,
                is_decoder,
                include_prev_tokens_mask=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, device, is_decoder
            )

        # *****************************************************************************
        # ************************ prepare cross-attention mask ************************
        # *****************************************************************************
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
            audio_length=audio_length
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentionsAndDisagreements(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            disagreements=encoder_outputs.disagreements
        )


class QformerLMHeadModel(BertPreTrainedModel, GenerationMixinHack):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = QformerModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            query_embeds=None,
            audio_embeds=None,
            audio_atts=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=True,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_logits=False,
            is_decoder=True,
            reduction="mean",
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None
            audio_embeds = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            audio_embeds=audio_embeds,
            # audio_atts=audio_atts,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )
        sequence_output = outputs[0]
        # get language output
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1]:, :]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        # calculate language modeling loss
        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        # make return output
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, query_embeds, audio_embeds=None, past_key_values=None,
            attention_mask=None, audio_atts=None,
            **model_kwargs
    ):
        """used in .generate"""
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        # if attention_mask is None:
        #     attention_mask = input_ids.new_ones(input_ids.shape)
        # query_mask = input_ids.new_ones(query_embeds.shape[:-1])
        # if audio_atts is not None:
        #     attention_mask = torch.cat([audio_atts, attention_mask], dim=-1)
        # attention_mask = torch.cat([query_mask, attention_mask], dim=-1)

        if attention_mask is None:
            att_mask_list = [input_ids.new_ones(query_embeds.shape[:-1])]
            if audio_atts is not None:
                att_mask_list.append(audio_atts)
            att_mask_list.append(input_ids.new_ones(input_ids.shape))
            attention_mask = torch.cat(att_mask_list, dim=-1)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "query_embeds": query_embeds,
            "audio_embeds": audio_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
