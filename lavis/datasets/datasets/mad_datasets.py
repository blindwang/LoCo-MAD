"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict, defaultdict
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset


class MadCaptionFeatureDatasetMixin:
    def collater(self, samples):
        collated_samples = defaultdict(list)
        for item in samples:
            for k, v in item.items():
                collated_samples[k].append(v)

        collated_samples_ts = {}
        for k, v in collated_samples.items():
            collated_feature = self.vis_processor.collate_feature(k, v)
            if type(collated_feature) is tuple:
                collated_feature, collated_mask = collated_feature
                collated_samples_ts[f"{k}_mask"] = collated_mask
            collated_samples_ts[k] = collated_feature
        return collated_samples_ts


class MadCaptionFeatureDataset(MadCaptionFeatureDatasetMixin, BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths
                 ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # self.feat_suffix = feat_suffix
        self.context_range = vis_processor.context_range
        self.char_num = vis_processor.char_num
        self.img_ids = {}
        n = 0
        # reorder img id
        for ann in self.annotation:
            img_id = ann["index"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.vid2gts = defaultdict(list)
        # {id: [cap1, cap2, ...]}
        # In MAD, each clip only has one cap
        # After data augmentation, we have two caps in some cases
        for ann in self.annotation:
            self.vid2gts[ann["index"]].append(ann["text"])
            if "aug" in list(ann.keys()):
                if ann["aug"] != "":
                    self.vid2gts[ann["index"]].append(ann["aug"])

    def __getitem__(self, index):

        ann = self.annotation[index]

        feat_name = ann["visual_feature_path"]  # movie-id_index_start-frame_end-frame.npz
        # vname = vname.replace(".mp4", self.feat_suffix)
        feature_path = os.path.join(self.vis_root, feat_name)

        # process visual, subtitle and text feature

        # context sub
        sub_context, cap_context = [], []
        context_feat_path = []
        if 'chars' not in ann:
            characters = []
        else:
            all_characters = self.annotation[index]['chars']
            characters = sorted(all_characters, key=lambda x: x[1], reverse=True)[:self.char_num]
            characters = [c[0] for c in characters]
        cur_movie = self.annotation[index]['movie']
        for i in range(self.context_range):
            if index - i - 1 >= 0:
                if self.annotation[index - i - 1]['movie'] == cur_movie:
                    # context ad
                    cap_context.append(self.annotation[index - i - 1]['text'])
                    # lang_feat_name = f"language_feature_unamed/{self.annotation[index - i - 1]['index']}.npz"
                    lang_feat_name = f"language_feature/{self.annotation[index - i - 1]['index']}.npz"
                    lang_feature_path = os.path.join(self.vis_root, lang_feat_name)
                    context_feat_path.append(lang_feature_path)
                    # context sub
                    if not self.annotation[index - i - 1]['subs']:
                        sub_context.append("")
                    else:
                        part_sub_data = [sub_data['text'] for sub_data in self.annotation[index - i - 1]['subs']]
                        sub_context.append(" ".join(part_sub_data))
                else:
                    context_feat_path.append("")
                    cap_context.append("")
                    sub_context.append("")
            else:
                context_feat_path.append("")
                cap_context.append("")
                sub_context.append("")
        sub_context.reverse()
        if not self.annotation[index]['subs']:
            sub_context.append("")
        else:
            cur_sub_data = [sub_data['text'] for sub_data in self.annotation[index]['subs']]
            sub_context.append(" ".join(cur_sub_data))
        cap_context.reverse()
        context_feat_path.reverse()
        # for i in range(self.context_range):
        #     if index + i + 1 < len(self.annotation):
        #         if self.annotation[index + i + 1]['movie'] == cur_movie:
        #             # context ad
        #             cap_context.append(self.annotation[index + i + 1]['text'])
        #             # lang_feat_name = f"language_feature_unamed/{self.annotation[index + i + 1]['index']}.npz"
        #             lang_feat_name = f"language_feature/{self.annotation[index + i + 1]['index']}.npz"
        #             lang_feature_path = os.path.join(self.vis_root, lang_feat_name)
        #             context_feat_path.append(lang_feature_path)
        #             # context sub
        #             if not self.annotation[index + i + 1]['subs']:
        #                 sub_context.append("")
        #             else:
        #                 part_sub_data = [sub_data['text'] for sub_data in self.annotation[index + i + 1]['subs']]
        #                 sub_context.append(" ".join(part_sub_data))
        #         else:
        #             context_feat_path.append("")
        #             cap_context.append("")
        #             sub_context.append("")
        #     else:
        #         context_feat_path.append("")
        #         cap_context.append("")
        #         sub_context.append("")

        feature_dict = self.vis_processor(feature_path, context_feat_path, sub_context, cap_context, characters)
        # gts relates to compute evaluation index, while caption is for training such as computing loss
        caption = self.text_processor(ann["text"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        data_item = {
            "text_input": caption,
            "image_id": ann["index"],
            "gts": self.vid2gts[ann["index"]],
        }
        # add visual feature
        data_item.update(feature_dict)  # feature_xxx example: feature_audio

        return data_item


class MadCaptionFeatureEvalDataset(MadCaptionFeatureDatasetMixin, BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # self.feat_suffix = feat_suffix
        self.context_range = vis_processor.context_range
        self.char_num = vis_processor.char_num
        self.img_ids = {}
        n = 0
        # reorder img id
        for ann in self.annotation:
            img_id = ann["index"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.vid2gts = defaultdict(list)
        # {id: [cap1, cap2, ...]}
        # In MAD, each clip only has one cap
        for ann in self.annotation:
            self.vid2gts[ann["index"]].append(ann["text"])

    def __getitem__(self, index):

        ann = self.annotation[index]

        feat_name = ann["visual_feature_path"]  # movie-id_index_start-frame_end-frame.npz
        # vname = vname.replace(".mp4", self.feat_suffix)
        feature_path = os.path.join(self.vis_root, feat_name)

        # process visual, subtitle and text feature

        # context sub
        sub_context, cap_context = [], []
        context_feat_path = []
        if 'chars' not in ann:
            characters = []
        else:
            all_characters = self.annotation[index]['chars']
            characters = sorted(all_characters, key=lambda x: x[1], reverse=True)[:self.char_num]
            characters = [c[0] for c in characters]
        cur_movie = self.annotation[index]['movie']
        for i in range(self.context_range):
            if index - i - 1>=0:
                if self.annotation[index - i - 1]['movie'] == cur_movie:
                    # context ad
                    cap_context.append(self.annotation[index - i - 1]['text'])
                    # lang_feat_name = f"language_feature_unamed_eval/{self.annotation[index - i - 1]['index']}.npz"
                    lang_feat_name = f"language_feature_eval/{self.annotation[index - i - 1]['index']}.npz"
                    lang_feature_path = os.path.join(self.vis_root, lang_feat_name)
                    context_feat_path.append(lang_feature_path)
                    # context sub
                    if not self.annotation[index - i - 1]['subs']:
                        sub_context.append("")
                    else:
                        part_sub_data = [sub_data['text'] for sub_data in self.annotation[index - i - 1]['subs']]
                        sub_context.append(" ".join(part_sub_data))
                else:
                    context_feat_path.append("")
                    cap_context.append("")
                    sub_context.append("")
            else:
                context_feat_path.append("")
                cap_context.append("")
                sub_context.append("")
        sub_context.reverse()
        if not self.annotation[index]['subs']:
            sub_context.append("")
        else:
            cur_sub_data = [sub_data['text'] for sub_data in self.annotation[index]['subs']]
            sub_context.append(" ".join(cur_sub_data))
        cap_context.reverse()
        context_feat_path.reverse()
        # for i in range(self.context_range):
        #     if index + i + 1<len(self.annotation):
        #         if self.annotation[index + i + 1]['movie'] == cur_movie:
        #             # context ad
        #             cap_context.append(self.annotation[index + i + 1]['text'])
        #             # lang_feat_name = f"language_feature_unamed_eval/{self.annotation[index + i + 1]['index']}.npz"
        #             lang_feat_name = f"language_feature_eval/{self.annotation[index + i + 1]['index']}.npz"
        #             lang_feature_path = os.path.join(self.vis_root, lang_feat_name)
        #             context_feat_path.append(lang_feature_path)
        #             # context sub
        #             if not self.annotation[index + i + 1]['subs']:
        #                 sub_context.append("")
        #             else:
        #                 part_sub_data = [sub_data['text'] for sub_data in self.annotation[index + i + 1]['subs']]
        #                 sub_context.append(" ".join(part_sub_data))
        #         else:
        #             context_feat_path.append("")
        #             cap_context.append("")
        #             sub_context.append("")
        #     else:
        #         context_feat_path.append("")
        #         cap_context.append("")
        #         sub_context.append("")

        # context ad
        # cap_context = []
        # for i in range(self.context_range):
        #     if index - i - 1 >= 0:
        #         if self.annotation[index - i - 1]['movie'] == cur_movie:
        #             cap_context.append(self.annotation[index - i - 1]['text'])
        # cap_context.reverse()
        # for i in range(self.context_range):
        #     if index + i + 1 < len(self.annotation):
        #         if self.annotation[index + i + 1]['movie'] == cur_movie:
        #             cap_context.append(self.annotation[index + i + 1]['text'])

        feature_dict = self.vis_processor(feature_path, context_feat_path, sub_context, cap_context, characters)
        # gts relates to compute evaluation index, while caption is for training such as computing loss
        caption = self.text_processor(ann["text"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        data_item = {
            "text_input": caption,
            "image_id": ann["index"],
            "gts": self.vid2gts[ann["index"]],
        }
        # add visual feature
        data_item.update(feature_dict)  # feature_xxx example: feature_audio

        return data_item