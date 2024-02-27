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


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["video"],
                "caption": ann["text_input"],
                "image": sample["image"],
            }
        )


class VideoCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": ann["image_id"],
        }


class VideoCaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


class VideoCaptionFeatureDatasetMixin:
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


class VideoCaptionFeatureDataset(VideoCaptionFeatureDatasetMixin, BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, feat_suffix=".npz",
                 ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.feat_suffix = feat_suffix

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.vid2gts = defaultdict(list)
        for ann in self.annotation:
            self.vid2gts[ann["image_id"]].append(ann["caption"])

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]  # xxx.mp4
        vname = vname.replace(".mp4", self.feat_suffix)
        feature_path = os.path.join(self.vis_root, vname)

        feature_dict = self.vis_processor(feature_path)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        data_item = {
            "text_input": caption,
            "image_id": ann["image_id"],
            "gts": self.vid2gts[ann["image_id"]],
        }
        data_item.update(feature_dict)  # feature_xxx example: feature_audio

        return data_item


class VideoCaptionFeatureEvalDataset(VideoCaptionFeatureDatasetMixin, BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, feat_suffix=".npz"):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.feat_suffix = feat_suffix

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.vid2gts = defaultdict(list)
        for ann in self.annotation:
            self.vid2gts[ann["image_id"]].append(ann["caption"])

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        vname = vname.replace(".mp4", self.feat_suffix)
        feature_path = os.path.join(self.vis_root, vname)

        feature_dict = self.vis_processor(feature_path)

        # "image_id" is kept to stay compatible with the COCO evaluation format
        data_item = {
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
            "gts": self.vid2gts[ann["image_id"]],
        }
        data_item.update(feature_dict)  # feature_xxx example: feature_audio

        return data_item
