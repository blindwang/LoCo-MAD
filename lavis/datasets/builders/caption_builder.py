"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
    VideoCaptionFeatureDataset,
    VideoCaptionFeatureEvalDataset
)

from lavis.datasets.datasets.mad_video_caption_datasets import (
    MadVideoCaptionFeatureDataset,
    MadVideoCaptionFeatureEvalDataset
)


@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }


@registry.register_builder("msrvtt_feature_caption")
class MSRVTTFeatureCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionFeatureDataset
    eval_dataset_cls = VideoCaptionFeatureEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/features_cap.yaml",
    }

@registry.register_builder("mad_feature_caption")
class MADFeatureCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = MadVideoCaptionFeatureDataset
    eval_dataset_cls = MadVideoCaptionFeatureEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mad/mad_caption.yaml",
    }