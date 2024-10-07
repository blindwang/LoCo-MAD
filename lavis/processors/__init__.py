"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.processors.base_processor import BaseProcessor

from lavis.processors.alpro_processors import (
    AlproVideoTrainProcessor,
    AlproVideoEvalProcessor,
)
from lavis.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from lavis.processors.gpt_processors import (
    GPTVideoFeatureProcessor,
    GPTDialogueProcessor,
)
from lavis.processors.clip_processors import ClipImageTrainProcessor

from lavis.processors.multimodal_feature_processors import (
    MMVCTVideoProcessor, VALQformerFeatureProcessor,
    VALQformerVideoTrainProcessor, VALQformerVideoEvalProcessor,
    VALQformerCaptionProcessor, VALQformerChCaptionProcessor
)

from lavis.processors.mad_multimodal_feature_processors import MADQformerFeatureProcessor
from lavis.processors.moviecaption_feature_processors import AutoADVideoTrainProcessor
from lavis.processors.moviecaption_feature_processors_recurrent import AutoADVideoTrainRecurrentProcessor

from lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    # ALPRO
    "AlproVideoTrainProcessor",
    "AlproVideoEvalProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    "ClipImageTrainProcessor",
    # GPT
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
    # MMVCT
    "MMVCTVideoProcessor",
    "VALQformerFeatureProcessor",
    "VALQformerVideoTrainProcessor",
    "VALQformerVideoEvalProcessor",
    "VALQformerCaptionProcessor",
    "VALQformerChCaptionProcessor",
    # MAD
    "AutoADVideoTrainProcessor",
    "MADQformerFeatureProcessor"
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
