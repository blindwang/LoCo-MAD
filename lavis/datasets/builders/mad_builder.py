"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.mad_datasets import MadCaptionFeatureDataset, MadCaptionFeatureEvalDataset


@registry.register_builder("mad_cap")
class MADBuilder(BaseDatasetBuilder):
    train_dataset_cls = MadCaptionFeatureDataset
    eval_dataset_cls = MadCaptionFeatureEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mad/defaults_caption.yaml",
        "eval": "configs/datasets/mad/defaults_caption.yaml",
    }