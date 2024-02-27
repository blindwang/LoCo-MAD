"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from .visual_audio_language_qformer import QformerBase, disabled_train
# from lavis.models.base_model import BaseModel
from .modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer

from transformers.modeling_utils import apply_chunking_to_forward
from einops import rearrange, repeat
from .init_dynamic_selection import init_dynamic_selection
import math


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


@registry.register_model("mad_qformer_opt")
class MADQformerOPT(QformerBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vitl": "configs/models/MADQformer/mad_inference_modeling.yaml",
    }

    def __init__(
            self,
            num_query_token=32,
            opt_model="facebook/opt-125m",
            prompt="According to the contextual subtitles and captions, describe the content of the clip",
            max_txt_len=32,
            contextual_max_len=256,
            # My dev: feature
            encoder_config: dict = None,
            # My dev: addition
            subtitle=True,
            caption=True,
            top_k=5,
            cross_attention_freq=1,
            num_hidden_layers=2,
            visual_num_temporal_embedding=None,
    ):
        """

        @param num_query_token:
        @param opt_model: pretrained language model name in hugging face
        @param prompt: trigger the LM
        @param max_txt_len: parameter for opt_tokenizer
        @param encoder_config: configuration of the encoder
        @param subtitle: whether to use the contextual subtitles
        @param sub_max_len: the max token length of contextual subtitles
        @param caption: whether to use the contextual captions
        @param cap_max_len: the max token length of contextual captions
        @param visual_num_temporal_embedding: number of the temporal embedding
        """
        super().__init__()

        self.subtitle = subtitle
        self.caption = caption
        self.contextual_max_len = contextual_max_len
        self.top_k = top_k
        self.visual_num_temporal_embedding = visual_num_temporal_embedding
        self.end2end = encoder_config['end2end']
        self.encoder_config = encoder_config
        self.reset_pretrain_queries = False
        self.discard_visual_cls_token = False
        self.visual_temporal_encoder_config = None
        self.query_temporal_embeddings = None

        if encoder_config['end2end'] is True:
            pass
            # self.visual_encoder, self.ln_vision = self.init_vision_encoder(**encoder_config)
            # self.dim_features = self.visual_encoder.num_features
            # if encoder_config.get('freeze_vit', True) is True:
            #     for name, param in self.visual_encoder.named_parameters():
            #         param.requires_grad = False
            #     self.visual_encoder = self.visual_encoder.eval()
            #     self.visual_encoder.train = disabled_train
            #     logging.info("freeze vision encoder")
            #
            # if audio is True:
            #     self.audio_max_len = audio_max_len
            #     self.audio_encoder, self.ln_audio = self.init_audio_encoder()
            #     if self.audio_encoder.config.hidden_size != 768:
            #         self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, 768)
            #     else:
            #         self.audio_proj = nn.Identity()
            #     for p in self.audio_encoder.parameters():
            #         p.requires_grad = False
        else:
            self.dim_features = encoder_config['dim_features']
            self.ln_vision = nn.LayerNorm(self.dim_features)
            # # audio
            # if audio:
            #     self.audio_max_len = audio_max_len
            #     if audio_dim_features != 768:  # Qformer is 768-d
            #         self.audio_proj = nn.Linear(audio_dim_features, 768)
            #     else:
            #         self.audio_proj = nn.Identity()
            #     self.ln_audio = nn.LayerNorm(audio_dim_features)

        self.tokenizer = self.init_tokenizer()

        # not able to handle more frames for now
        if visual_num_temporal_embedding is not None:
            logging.info('activate num_temporal_embedding')
            self.pos_encoding = VanillaPositionalEncoding(self.dim_features, visual_num_temporal_embedding)
            # self.visual_temporal_embeddings = nn.ParameterList(
            #     nn.Parameter(torch.zeros(self.dim_features))
            #     for _ in range(visual_num_temporal_embedding)
            # )
        else:
            self.visual_temporal_embeddings = None

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.dim_features,
            # audio=audio, audio_max_len=audio_max_len
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        # add special tokens
        new_tokens_list = ["BSB_token", "ESB_token", "BCP_token", "ECP_token", "VB_token", "VE_token"]
        num_added_tokens = self.opt_tokenizer.add_tokens(new_tokens_list)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer))

        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt

        # cross attention layer for dynamic contextual text selection
        if self.top_k > 0:
            # self.cross_attention = nn.MultiheadAttention(
            #     embed_dim=self.opt_model.config.hidden_size, num_heads=8, dropout=0.1, batch_first=True, device=self.opt_model.device
            # )
            self.dynamic_selection = init_dynamic_selection(self.dim_features, cross_attention_freq,
                                                            num_hidden_layers)
            self.linear = nn.Linear(self.dim_features, 1, device=self.opt_model.device)
            self.softmax = nn.Softmax(dim=-1)
            self.cross_entropy = nn.CrossEntropyLoss()
            self.temp = nn.Parameter(0.05 * torch.ones([]))
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        if self.end2end is True:
            pass
            # with self.maybe_autocast():
            #     samples['feature_visual'] = self._prepare_visual_feature(samples['video'])
            #     del samples['video']
            #     if self.audio:
            #         feature_audio = self._prepare_audio_feature(samples['audio'], samples['audio_mask'])
            #         samples['feature_audio'], samples['feature_audio_mask'] = feature_audio
            #         del samples['audio'], samples['audio_mask']

        video_embeds, video_atts = self._prepare_video_embeds(
            samples['feature_visual'], samples['feature_visual_mask'],
            visual_sp_query_feature=samples.get('visual_sp_query_feature', None)
        )
        # audio_embeds, audio_atts = None, None
        sub_tokens, cap_tokens, sub_cap_tokens, prompt_tokens = None, None, None, None
        device = video_embeds.device
        # dynamic contextual text selection
        select_loss = 0
        topk_relavency_score, topk_relavency_index = None, None
        if self.top_k > 0:
            text_embeds = samples["feature_language"]
            text_atts = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(device)
            # cross-attention : contextual text (Q) and visual features (K, V)
            dynamic_selection_output = self.dynamic_selection(hidden_states=text_embeds,
                                                              attention_mask=text_atts,
                                                              encoder_hidden_states=video_embeds,
                                                              encoder_attention_mask=video_atts)
            # compute the relavency score based on attention output
            linear_output = self.linear(dynamic_selection_output.last_hidden_state).squeeze(-1)
            relavency_score = self.softmax(linear_output)
            # pick the top-k relavency score in a differentiable way
            topk_relavency_score, topk_relavency_index = torch.topk(relavency_score, self.top_k)

            # get the top-k contextual text embedding
            selected_text_embeds = torch.gather(text_embeds, 1,
                                                topk_relavency_index.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1)))
            non_selected_indices = torch.ones((text_embeds.shape[0], text_embeds.shape[1]),
                                              dtype=torch.bool,
                                              device=text_embeds.device)
            non_selected_indices.scatter_(1, topk_relavency_index, 0)
            non_selected_text_embeds = torch.masked_select(text_embeds,
                                                           non_selected_indices.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1)))
            non_selected_text_embeds = non_selected_text_embeds.reshape(text_embeds.shape[0], -1, text_embeds.size(-1))
            # selected_text_embeds_list, non_selected_text_embeds_list = [], []
            # for i in range(text_embeds.shape[0]):
            #     selected_text_embeds_list.append(text_embeds[i, topk_relavency_index[i]])
            #     non_selected_text_embeds_list.append(text_embeds[i, [j for j in range(text_embeds.shape[1])
            #                                                          if j not in topk_relavency_index[i]]])
            # selected_text_embeds, non_selected_text_embeds = torch.stack(selected_text_embeds_list), torch.stack(
            #     non_selected_text_embeds_list)
            # contrastive loss
            visual_feature = torch.mean(video_embeds,
                                        dim=1)  # video_embeds: [batch, n_frame, embed_dim]
            selected_text_feature = torch.sum(selected_text_embeds * topk_relavency_score.unsqueeze(-1),
                                              dim=1)  # selected_text_embeds: [batch, topk_n, embed_dim]
            pos = torch.cosine_similarity(visual_feature, selected_text_feature)
            # neg = torch.stack([torch.cosine_similarity(visual_feature, non_selected_text_embeds[:, i, :])
            #                    for i in range(non_selected_text_embeds.shape[1])], dim=1)
            neg = torch.cosine_similarity(
                visual_feature.unsqueeze(1),  # unsqueeze to match dimensions
                non_selected_text_embeds,
                dim=2
            )
            logits = torch.cat([pos.unsqueeze(1), neg], dim=1) / self.temp
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            select_loss = self.cross_entropy(logits, labels)
            # # compute global text feature based on all contextual text embedding, use relavency_score as weights
            # global_text_feature = torch.mean(text_embeds, dim=1)
            # # compute local text feature based on selected contextual text embedding, use topk_relavency_score as weights
            # local_text_feature = torch.mean(selected_text_embeds, dim=1)
            # # compute the similarity between global text feature and local text feature
            # similarity = torch.cosine_similarity(global_text_feature, local_text_feature, dim=-1)
            # # normalize to [0, 1]
            # select_loss = (1 - torch.mean(similarity)) / 2

        if self.subtitle and self.caption:
            # all_contextual_text_attn_output, all_contextual_text_attn_output_weights = None, None
            # all_contextual_text_embeds = None
            # compute cross-attention between j-th contextual text (Q) and visual features (K, V)
            # for j in range(len(samples['caption'][0])):
            #     samples_sub_and_cap = []
            #     # i-th sample (total as one batch)
            #     assert len(samples['caption']) == len(samples['subtitle'])
            #     for i in range(len(samples['caption'])):
            #         sample_sub_cap = ""
            #         sample_sub_cap += "BSB_token" + samples['subtitle'][i][j] + "ESB_token"
            #         sample_sub_cap += "BCP_token" + samples['caption'][i][j] + "ECP_token"
            #         samples_sub_and_cap.append(sample_sub_cap)
            #     samples_sub_and_cap_token = self.opt_tokenizer(samples_sub_and_cap,
            #                                 return_tensors="pt",
            #                                 truncation=True,
            #                                 padding="max_length",
            #                                 max_length=self.contextual_max_len,
            #                                 add_special_tokens = False
            #                                 ).to(device)
            #     text_embeds = self.opt_model.model.decoder.embed_tokens(samples_sub_and_cap_token['input_ids'])
            #     if all_contextual_text_embeds is None:
            #         all_contextual_text_embeds = text_embeds.unsqueeze(1)
            #     else:
            #         all_contextual_text_embeds = torch.cat([all_contextual_text_embeds, text_embeds.unsqueeze(1)], dim=1)
            #     # cross-attention : contextual text (Q) and visual features (K, V)
            #     jth_contextual_text_attn_output, jth_contextual_text_attn_output_weights = self.cross_attention(
            #     query=text_embeds, key=video_embeds, value=video_embeds)
            #     jth_contextual_text_attn_output = jth_contextual_text_attn_output.unsqueeze(1)
            #     jth_contextual_text_attn_output_weights = jth_contextual_text_attn_output_weights.unsqueeze(1)
            #     if all_contextual_text_attn_output is None:
            #         all_contextual_text_attn_output = jth_contextual_text_attn_output
            #         all_contextual_text_attn_output_weights = jth_contextual_text_attn_output_weights
            #     else:
            #         all_contextual_text_attn_output = torch.cat([all_contextual_text_attn_output, jth_contextual_text_attn_output], dim=1)
            #         all_contextual_text_attn_output_weights = torch.cat([all_contextual_text_attn_output_weights, jth_contextual_text_attn_output_weights], dim=1)
            # batchsize, sequence_len, token_num, dim_num = all_contextual_text_attn_output.size()
            # attn_output = all_contextual_text_attn_output.view(batchsize, sequence_len, token_num * dim_num)
            # compute the relavency score based on attention output
            # linear_output = self.linear(attn_output).squeeze(-1)
            # relavency_score = self.softmax(linear_output)

            # topk_relavency_index = torch.multinomial(relavency_score, self.top_k)
            # most_relative_index = torch.argmax(relavency_score, dim=1)
            # selected_text_embeds = all_contextual_text_embeds[:, most_relative_index, :]
            # get the top-k contextual text embedding
            # add special token for subtitles and captions
            subs_and_caps = []
            for i in range(len(samples['caption'])):
                sample_sub_cap = ""
                assert len(samples['caption'][i]) == len(
                    samples['subtitle'][i]), "subtitle list and caption list should have the same length"
                context_range = len(samples['caption'][i])
                contexture_index = [j for j in range(max(0,context_range//2-2), min(context_range,context_range//2+4))]
                for j in range(len(samples['caption'][i])):
                    if topk_relavency_index is not None:
                        if j not in topk_relavency_index[i] and j not in contexture_index:
                            continue
                    if samples['subtitle'][i][j] != "":
                        sample_sub_cap += "BSB_token" + samples['subtitle'][i][j] + "ESB_token"
                    if samples['caption'][i][j] != "":
                        sample_sub_cap += "BCP_token" + samples['caption'][i][j] + "ECP_token"
                subs_and_caps.append(sample_sub_cap)
            sub_cap_tokens = self.opt_tokenizer(subs_and_caps,
                                                return_tensors="pt",
                                                truncation=True,
                                                padding=True,
                                                max_length=self.contextual_max_len,
                                                add_special_tokens=False
                                                ).to(device)

        elif self.subtitle:
            subs = []
            for i in range(len(samples['caption'])):
                sample_sub = ""
                context_range = len(samples['caption'][i])
                contexture_index = [j for j in
                                    range(max(0, context_range // 2 - 2), min(context_range, context_range // 2 + 4))]
                for j in range(len(samples['caption'][i])):
                    if topk_relavency_index is not None:
                        if j not in topk_relavency_index[i] and j not in contexture_index:
                            continue
                    if samples['subtitle'][i][j] != "":
                        sample_sub += "BSB_token" + samples['subtitle'][i][j] + "ESB_token"
                subs.append(sample_sub)
            sub_tokens = self.opt_tokenizer(subs,
                                            return_tensors="pt",
                                            truncation=True,
                                            padding=True,
                                            max_length=self.contextual_max_len,
                                            add_special_tokens=False
                                            ).to(device)
            # sub_begin_token = self.opt_tokenizer("BSB_token", return_tensors="pt", add_special_tokens=True).to(device)
            # sub_end_token = self.opt_tokenizer("ESB_token", return_tensors="pt", add_special_tokens=True).to(device)
            # for sub in samples['subtitle']:
            #     one_sub_tokens = self.opt_tokenizer(' '.join(sub),
            #                                         return_tensors="pt",
            #                                         truncation=True,
            #                                         padding=True,
            #                                         max_length=self.sub_max_len
            #                                         ).to(device)
            #     one_sub_tokens = torch.cat([sub_begin_token.input_ids, one_sub_tokens.input_ids, sub_end_token.input_ids], dim=1)
            #     if sub_tokens is not None:
            #         sub_tokens = torch.cat([sub_tokens, one_sub_tokens], dim=0)
            #     else:
            #         sub_tokens = one_sub_tokens

        elif self.caption:
            caps = []
            for i in range(len(samples['caption'])):
                sample_cap = ""
                context_range = len(samples['caption'][i])
                contexture_index = [j for j in
                                    range(max(0, context_range // 2 - 2), min(context_range, context_range // 2 + 4))]
                for j in range(len(samples['caption'][i])):
                    if topk_relavency_index is not None:
                        if j not in topk_relavency_index[i] and j not in contexture_index:
                            continue
                    if samples['caption'][i][j] != "":
                        sample_cap += "BCP_token" + samples['caption'][i][j] + "ECP_token"
                caps.append(sample_cap)
            cap_tokens = self.opt_tokenizer(caps,
                                            return_tensors="pt",
                                            truncation=True,
                                            padding=True,
                                            max_length=self.contextual_max_len,
                                            add_special_tokens=False
                                            ).to(device)
        #     # add special token for captions
        #     cap_begin_token = self.opt_tokenizer("BCP_token", return_tensors="pt", add_special_tokens=True).to(device)
        #     cap_end_token = self.opt_tokenizer("ECP_token", return_tensors="pt", add_special_tokens=True).to(device)
        #     for cap in samples['caption']:
        #         one_cap_tokens = self.opt_tokenizer(' '.join(cap),
        #                                             return_tensors="pt",
        #                                             truncation=True,
        #                                             padding=True,
        #                                             max_length=self.cap_max_len
        #                                             ).to(device)
        #         one_cap_tokens = torch.cat([cap_begin_token.input_ids, one_cap_tokens.input_ids, cap_end_token.input_ids], dim=1)
        #         if cap_tokens is not None:
        #             cap_tokens = torch.cat([cap_tokens, one_cap_tokens], dim=0)
        #         else:
        #             cap_tokens = one_cap_tokens

        if self.prompt:
            prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt").to(device)

        # if self.audio:
        #     audio_embeds, audio_atts = self._prepare_audio_embeds(
        #         samples['feature_audio'], samples['feature_audio_mask']
        #     )

        device = video_embeds.device
        bs = video_embeds.shape[0]

        attention_mask = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(device)
        # if audio_atts is not None:
        #     attention_mask = torch.cat([attention_mask, audio_atts], dim=1)

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        # if self.query_temporal_embeddings is not None:
        #     query_tokens = self._add_te_to_query_tokens(query_tokens)
        video_mix_output = self.Qformer.bert(
            query_embeds=query_tokens,
            # audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=video_embeds,  # cross-attn
            encoder_attention_mask=video_atts,  # cross-attn
            return_dict=True,
        )

        vid_tokens = self.opt_proj(video_mix_output.last_hidden_state)
        # add special token for video
        begin_video_token = self.opt_tokenizer("VB_token",return_tensors="pt",add_special_tokens=False).to(device)
        end_video_token = self.opt_tokenizer("VE_token",return_tensors="pt",add_special_tokens=False).to(device)
        begin_video_embed = self.opt_model.model.decoder.embed_tokens(begin_video_token.input_ids).expand(bs, -1, -1)
        end_video_embed = self.opt_model.model.decoder.embed_tokens(end_video_token.input_ids).expand(bs, -1, -1)
        if begin_video_embed.shape[-1] != vid_tokens.shape[-1]:
            begin_video_embed = self.opt_model.model.decoder.project_in(begin_video_embed)
            end_video_embed = self.opt_model.model.decoder.project_in(end_video_embed)
        vid_tokens = torch.cat([begin_video_embed, vid_tokens, end_video_embed], dim=1)
        prompts, prompt_atts = self.prompt_wrap(vid_tokens, sub_cap_tokens, sub_tokens, cap_tokens, prompt_tokens)

        self.opt_tokenizer.padding_side = "right"
        text = [t + "\n" for t in samples["text_input"]]
        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        # if self.prompt:
        #     targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
        empty_targets = (
            torch.ones(prompts.size()[:-1], dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        if inputs_embeds.shape[-1] != vid_tokens.shape[-1]:
            inputs_embeds = self.opt_model.model.decoder.project_in(inputs_embeds)
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        attention_mask = torch.cat([prompt_atts, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss + select_loss

        return {"loss": loss, "select_loss": select_loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # image = samples["image"]
        with self.maybe_autocast():
            if self.end2end is True:
                pass
                # samples['feature_visual'] = self._prepare_visual_feature(samples['video'])
                # del samples['video']
                # if self.audio:
                #     feature_audio = self._prepare_audio_feature(samples['audio'], samples['audio_mask'])
                #     samples['feature_audio'], samples['feature_audio_mask'] = feature_audio
                #     del samples['audio'], samples['audio_mask']

            video_embeds, video_atts = self._prepare_video_embeds(
                samples['feature_visual'], samples['feature_visual_mask'],
                visual_sp_query_feature=samples.get('visual_sp_query_feature', None)
            )
            # audio_embeds, audio_atts = None, None
            # if self.audio:
            #     audio_embeds, audio_atts = self._prepare_audio_embeds(
            #         samples['feature_audio'], samples['feature_audio_mask']
            #     )

            sub_tokens, cap_tokens, sub_cap_tokens, prompt_tokens = None, None, None, None
            device = video_embeds.device
            # cross-attention : contextual text (Q) and visual features (K, V)

            # if self.subtitle and self.caption:
            #     # add special token for subtitles and captions
            #     subs_and_caps = []
            #     for i in range(len(samples['caption'])):
            #         sample_sub_cap = ""
            #         assert len(samples['caption'][i]) == len(
            #             samples['subtitle'][i]), "subtitle list and caption list should have the same length"
            #         for j in range(len(samples['caption'][i])):
            #             sample_sub_cap += "BSB_token" + samples['subtitle'][i][j] + "ESB_token"
            #             sample_sub_cap += "BCP_token" + samples['caption'][i][j] + "ECP_token"
            #         subs_and_caps.append(sample_sub_cap)
            #     sub_cap_tokens = self.opt_tokenizer(subs_and_caps,
            #                                         return_tensors="pt",
            #                                         truncation=True,
            #                                         padding=True,
            #                                         add_special_tokens=False
            #                                         ).to(device)
            # dynamic contextual text selection
            topk_relavency_score, topk_relavency_index = None, None
            if self.top_k > 0:
                text_embeds = samples["feature_language"]
                text_atts = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(device)
                # cross-attention : contextual text (Q) and visual features (K, V)
                dynamic_selection_output = self.dynamic_selection(hidden_states=text_embeds,
                                                                  attention_mask=text_atts,
                                                                  encoder_hidden_states=video_embeds,
                                                                  encoder_attention_mask=video_atts, )
                # compute the relavency score based on attention output
                linear_output = self.linear(dynamic_selection_output.last_hidden_state).squeeze(-1)
                relavency_score = self.softmax(linear_output)
                # pick the top-k relavency score in a differentiable way
                topk_relavency_score, topk_relavency_index = torch.topk(relavency_score, self.top_k)

            if self.subtitle and self.caption:
                # all_contextual_text_attn_output, all_contextual_text_attn_output_weights = None, None
                # all_contextual_text_embeds = None
                # # compute cross-attention between j-th contextual text (Q) and visual features (K, V)
                # for j in range(len(samples['caption'][0])):
                #     samples_sub_and_cap = []
                #     # i-th sample (total as one batch)
                #     assert len(samples['caption']) == len(samples['subtitle'])
                #     for i in range(len(samples['caption'])):
                #         sample_sub_cap = ""
                #         sample_sub_cap += "BSB_token" + samples['subtitle'][i][j] + "ESB_token"
                #         sample_sub_cap += "BCP_token" + samples['caption'][i][j] + "ECP_token"
                #         samples_sub_and_cap.append(sample_sub_cap)
                #     samples_sub_and_cap_token = self.opt_tokenizer(samples_sub_and_cap,
                #                                                    return_tensors="pt",
                #                                                    truncation=True,
                #                                                    padding="max_length",
                #                                                    max_length=self.contextual_max_len,
                #                                                    add_special_tokens=False
                #                                                    ).to(device)
                #     text_embeds = self.opt_model.model.decoder.embed_tokens(samples_sub_and_cap_token['input_ids'])
                #     if all_contextual_text_embeds is None:
                #         all_contextual_text_embeds = text_embeds.unsqueeze(1)
                #     else:
                #         all_contextual_text_embeds = torch.cat([all_contextual_text_embeds, text_embeds.unsqueeze(1)],
                #                                                dim=1)
                #     # cross-attention : contextual text (Q) and visual features (K, V)
                #     jth_contextual_text_attn_output, jth_contextual_text_attn_output_weights = self.cross_attention(
                #         query=text_embeds, key=video_embeds, value=video_embeds)
                #     jth_contextual_text_attn_output = jth_contextual_text_attn_output.unsqueeze(1)
                #     jth_contextual_text_attn_output_weights = jth_contextual_text_attn_output_weights.unsqueeze(1)
                #     if all_contextual_text_attn_output is None:
                #         all_contextual_text_attn_output = jth_contextual_text_attn_output
                #         all_contextual_text_attn_output_weights = jth_contextual_text_attn_output_weights
                #     else:
                #         all_contextual_text_attn_output = torch.cat(
                #             [all_contextual_text_attn_output, jth_contextual_text_attn_output], dim=1)
                #         all_contextual_text_attn_output_weights = torch.cat(
                #             [all_contextual_text_attn_output_weights, jth_contextual_text_attn_output_weights], dim=1)
                #
                # batchsize, sequence_len, token_num, dim_num = all_contextual_text_attn_output.size()
                # attn_output = all_contextual_text_attn_output.view(batchsize, sequence_len, token_num * dim_num)
                # # compute the relavency score based on attention output
                # linear_output = self.linear(attn_output).squeeze(-1)
                # relavency_score = self.softmax(linear_output)
                # # pick the top-k relavency score in a differentiable way
                # topk_relavency_score, topk_relavency_index = torch.topk(relavency_score, self.top_k)
                # # topk_relavency_index = torch.multinomial(relavency_score, self.top_k)
                # # most_relative_index = torch.argmax(relavency_score, dim=1)
                # # selected_text_embeds = all_contextual_text_embeds[:, most_relative_index, :]
                # # get the top-k contextual text embedding
                # selected_text_embeds = torch.gather(all_contextual_text_embeds, 2,
                #                                     topk_relavency_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,
                #                                                                                             token_num,
                #                                                                                             dim_num))
                # selected_text_embeds = selected_text_embeds.view(batchsize, -1, dim_num)
                subs_and_caps = []
                for i in range(len(samples['caption'])):
                    sample_sub_cap = ""
                    assert len(samples['caption'][i]) == len(
                        samples['subtitle'][i]), "subtitle list and caption list should have the same length"
                    context_range = len(samples['caption'][i])
                    contexture_index = [j for j in
                                        range(max(0, context_range // 2 - 2),
                                              min(context_range, context_range // 2 + 4))]
                    for j in range(len(samples['caption'][i])):
                        if topk_relavency_index is not None:
                            if j not in topk_relavency_index[i] and j not in contexture_index:
                                continue
                        if samples['subtitle'][i][j] != "":
                            sample_sub_cap += "BSB_token" + samples['subtitle'][i][j] + "ESB_token"
                        if samples['caption'][i][j] != "":
                            sample_sub_cap += "BCP_token" + samples['caption'][i][j] + "ECP_token"
                    subs_and_caps.append(sample_sub_cap)
                sub_cap_tokens = self.opt_tokenizer(subs_and_caps,
                                                    return_tensors="pt",
                                                    truncation=True,
                                                    padding=True,
                                                    max_length=self.contextual_max_len,
                                                    add_special_tokens=False
                                                    ).to(device)

            elif self.subtitle:
                subs = []
                for i in range(len(samples['caption'])):
                    sample_sub = ""
                    context_range = len(samples['caption'][i])
                    contexture_index = [j for j in
                                        range(max(0, context_range // 2 - 2),
                                              min(context_range, context_range // 2 + 4))]
                    for j in range(len(samples['caption'][i])):
                        if topk_relavency_index is not None:
                            if j not in topk_relavency_index[i] and j not in contexture_index:
                                continue
                        if samples['subtitle'][i][j] != "":
                            sample_sub += "BSB_token" + samples['subtitle'][i][j] + "ESB_token"
                    subs.append(sample_sub)
                sub_tokens = self.opt_tokenizer(subs,
                                                return_tensors="pt",
                                                truncation=True,
                                                padding=True,
                                                max_length=self.contextual_max_len,
                                                add_special_tokens=False
                                                ).to(device)

            elif self.caption:
                caps = []
                for i in range(len(samples['caption'])):
                    sample_cap = ""
                    context_range = len(samples['caption'][i])
                    contexture_index = [j for j in
                                        range(max(0, context_range // 2 - 2),
                                              min(context_range, context_range // 2 + 4))]
                    for j in range(len(samples['caption'][i])):
                        if topk_relavency_index is not None:
                            if j not in topk_relavency_index[i] and j not in contexture_index:
                                continue
                        if samples['caption'][i][j] != "":
                            sample_cap += "BCP_token" + samples['caption'][i][j] + "ECP_token"
                    caps.append(sample_cap)
                cap_tokens = self.opt_tokenizer(caps,
                                                return_tensors="pt",
                                                truncation=True,
                                                padding=True,
                                                max_length=self.contextual_max_len,
                                                add_special_tokens=False
                                                ).to(device)

            if self.prompt:
                prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt").to(device)

            device = video_embeds.device
            bs = video_embeds.shape[0]

            attention_mask = torch.ones((bs, self.query_tokens.shape[1]), dtype=torch.long).to(device)
            # if audio_atts is not None:
            #     attention_mask = torch.cat([attention_mask, audio_atts], dim=1)

            query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
            if self.query_temporal_embeddings is not None:
                query_tokens = self._add_te_to_query_tokens(query_tokens)
            video_mix_output = self.Qformer.bert(
                query_embeds=query_tokens,
                # audio_embeds=audio_embeds,
                attention_mask=attention_mask,
                encoder_hidden_states=video_embeds,  # cross-attn
                encoder_attention_mask=video_atts,  # cross-attn
                return_dict=True,
            )

            vid_tokens = self.opt_proj(video_mix_output.last_hidden_state)
            # add special token for video
            begin_video_token = self.opt_tokenizer("VB_token", return_tensors="pt", add_special_tokens=False).to(device)
            end_video_token = self.opt_tokenizer("VE_token", return_tensors="pt", add_special_tokens=False).to(device)
            begin_video_embed = self.opt_model.model.decoder.embed_tokens(begin_video_token.input_ids).expand(bs, -1,
                                                                                                              -1)
            end_video_embed = self.opt_model.model.decoder.embed_tokens(end_video_token.input_ids).expand(bs, -1, -1)
            if begin_video_embed.shape[-1] != vid_tokens.shape[-1]:
                begin_video_embed = self.opt_model.model.decoder.project_in(begin_video_embed)
                end_video_embed = self.opt_model.model.decoder.project_in(end_video_embed)
            vid_tokens = torch.cat([begin_video_embed, vid_tokens, end_video_embed], dim=1)
            prompts, prompt_atts = self.prompt_wrap(vid_tokens, sub_cap_tokens, sub_tokens, cap_tokens, prompt_tokens)

            # if "prompt" in samples.keys():
            #     prompt = samples["prompt"]
            # else:
            #     prompt = self.prompt
            #
            # prompt = [prompt] * bs

            # opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(device)
            # input_ids = opt_tokens.input_ids
            # attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                prompts = prompts.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                prompts = prompts.repeat_interleave(num_beams, dim=0)

            input_ids = (
                torch.LongTensor(bs, 1)
                .fill_(self.opt_tokenizer(":", add_special_tokens=False).input_ids[0])
                .to(device)
            )
            # attention_mask = torch.cat([
            #     prompt_atts,
            #     torch.ones([bs, 1], dtype=torch.long, device=device)
            # ], dim=1)

            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=prompts[:, :-1],
                # inputs_embeds=prompts,
                attention_mask=prompt_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            # prompt_length = prompts.shape[1]
            prompt_length = 1
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")
        prompt = cfg.get("prompt", "")

        encoder_config = cfg.get("encoder_config", {'end2end': False, 'dim_features': 1024})
        subtitle = cfg.get("subtitle")
        caption = cfg.get("caption")
        top_k = cfg.get("top_k", 0)
        contextual_max_len = cfg.get("contextual_max_len", 256)
        visual_num_temporal_embedding = cfg.get("visual_num_temporal_embedding", None)
        num_hidden_layers = cfg.get("num_hidden_layers", 2)

        model = cls(
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            encoder_config=encoder_config,
            caption=caption,
            subtitle=subtitle,
            contextual_max_len=contextual_max_len,
            top_k=top_k,
            visual_num_temporal_embedding=visual_num_temporal_embedding,
            cross_attention_freq=1,
            num_hidden_layers=num_hidden_layers
        )

        model.load_checkpoint_from_config(cfg)

        return model

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

    # def _prepare_audio_feature(self, audio, audio_mask):
    #     # audio: B, T, c1, c2
    #     bs, t = audio.shape[0], audio.shape[1]
    #     # assert (bs * t) % 4 == 0, f"Please make (batch_size * temporal) % 4 == 0"
    #     audio = rearrange(audio, 'b t c1 c2 -> (b t) c1 c2')
    #
    #     audio_feature = apply_chunking_to_forward(
    #         lambda x: self.audio_encoder(input_values=x).pooler_output,  # B, 768
    #         bs * t // 4,
    #         0,
    #         audio
    #     )
    #     # audio_feature = self.audio_encoder(input_values=audio).pooler_output
    #     audio_feature = rearrange(audio_feature, '(b t) c -> b t c', b=bs, t=t)
    #     return audio_feature, audio_mask

    def _prepare_video_embeds(self, visual_feature: torch.Tensor, feature_visual_mask: torch.Tensor,
                              visual_sp_query_feature=None):
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

    # def _prepare_audio_embeds(self, audio_feature, audio_atts):
    #     # batch size, temporal, feature dim
    #     b, t, c = audio_feature.shape
    #
    #     # inject temporal information
    #     # assert t <= self.audio_max_len, "input audio frames too much"
    #     # temp_emb = torch.stack([self.audio_temporal_embeddings[i] for i in range(t)], dim=0)  # t, c
    #     # audio_feature += repeat(temp_emb, 't c -> b t c', b=b)
    #
    #     audio_embeds = self.audio_proj(self.ln_audio(audio_feature))
    #
    #     return audio_embeds, audio_atts

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

    def prompt_wrap(self, video_tokens, sub_cap_tokens, sub_tokens, cap_tokens, prompt_tokens):
        device = video_tokens.device
        bs = video_tokens.shape[0]
        wrapped_prompt = video_tokens
        wrapped_prompt_atts = torch.ones(wrapped_prompt.size()[:-1], dtype=torch.long, device=device)
        # if selected_text_embeds is not None:
        #     wrapped_prompt = torch.cat([wrapped_prompt, selected_text_embeds], dim=1)
        if sub_cap_tokens is not None:
            sub_cap_embeds = self.opt_model.model.decoder.embed_tokens(sub_cap_tokens.input_ids.long()).expand(bs, -1,
                                                                                                               -1)
            if sub_cap_embeds.shape[-1] != wrapped_prompt.shape[-1]:
                sub_cap_embeds = self.opt_model.model.decoder.project_in(sub_cap_embeds)
            wrapped_prompt = torch.cat([wrapped_prompt, sub_cap_embeds], dim=1)
            wrapped_prompt_atts = torch.cat([wrapped_prompt_atts, sub_cap_tokens.attention_mask], dim=1)
        if sub_tokens is not None:
            sub_embeds = self.opt_model.model.decoder.embed_tokens(sub_tokens.input_ids.long()).expand(bs, -1, -1)
            if sub_embeds.shape[-1] != wrapped_prompt.shape[-1]:
                sub_embeds = self.opt_model.model.decoder.project_in(sub_embeds)
            wrapped_prompt = torch.cat([wrapped_prompt, sub_embeds], dim=1)
            wrapped_prompt_atts = torch.cat([wrapped_prompt_atts, sub_tokens.attention_mask], dim=1)
        if cap_tokens is not None:
            cap_embeds = self.opt_model.model.decoder.embed_tokens(cap_tokens.input_ids.long()).expand(bs, -1, -1)
            if cap_embeds.shape[-1] != wrapped_prompt.shape[-1]:
                cap_embeds = self.opt_model.model.decoder.project_in(cap_embeds)
            wrapped_prompt = torch.cat([wrapped_prompt, cap_embeds], dim=1)
            wrapped_prompt_atts = torch.cat([wrapped_prompt_atts, cap_tokens.attention_mask], dim=1)
        if prompt_tokens is not None:
            prompt_embeds = self.opt_model.model.decoder.embed_tokens(prompt_tokens.input_ids.long()).expand(bs, -1, -1)
            if prompt_embeds.shape[-1] != wrapped_prompt.shape[-1]:
                prompt_embeds = self.opt_model.model.decoder.project_in(prompt_embeds)
            wrapped_prompt = torch.cat([wrapped_prompt, prompt_embeds], dim=1)
            wrapped_prompt_atts = torch.cat([wrapped_prompt_atts, prompt_tokens.attention_mask.expand(bs, -1)], dim=1)
        return wrapped_prompt, wrapped_prompt_atts
