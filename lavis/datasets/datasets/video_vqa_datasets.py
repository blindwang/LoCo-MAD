"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict, defaultdict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from lavis.datasets.datasets.base_dataset import BaseDataset


class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _build_class_labels(self, ans_path):
        ans2label = json.load(open(ans_path))

        self.class_labels = ans2label

    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(vpath)
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }


class VideoQAFeatureDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, feat_suffix=".npz",
                 ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.feat_suffix = feat_suffix

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]  # xxx.mp4
        vname = vname.replace(".mp4", self.feat_suffix)
        feature_path = os.path.join(self.vis_root, vname)

        feature_dict = self.vis_processor(feature_path)
        question = self.text_processor(ann["question"])

        data_item = {
            "text_input": question,
            "answers": ann["answer"],
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
        data_item.update(feature_dict)  # feature_xxx example: feature_audio

        return data_item

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
