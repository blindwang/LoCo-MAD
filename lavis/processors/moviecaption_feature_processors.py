import logging

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors import transforms_video
from lavis.processors.randaugment import VideoRandomAugment
from lavis.processors import functional_video as F
from torchvision import transforms
from lavis.datasets.data_utils import load_video

import soundfile
import resampy
from omegaconf import OmegaConf
import subprocess
import tempfile
from collections import defaultdict
from transformers import AutoProcessor, BatchFeature

from typing import List
from torch import Tensor
from pathlib import Path
import re

MAX_INT = registry.get("MAX_INT")


@registry.register_processor("auto_ad_video_train")
class AutoADVideoTrainProcessor(BaseProcessor):
    def __init__(
            self,
            # input_template,
            # output_keys,
            pad_key_lens=None,
            # missing_mod_default=None,
            context_range=10,
    ):
        """
        Args:
            # input_template (list):
            #     List of template string `folder_name.suffix`
            #     Examples: eva_vit_G.npz
            # output_keys (list):
            #     List of output keys in string. The order corresponds to `input_template`
            pad_key_lens (list):
                List of key-length tuple. `Key` is the output feature key which needs padded,
                `length` is the pad length.
                 Examples: ('feature_visual', 12), ('feature_audio', 8)
            # missing_mod_default (list):
            #     List of key-dims tuple. `Key` is the output feature key which may missing,
            #     `dims` are the default dimension.
            #     Examples: ('feature_audio', 0, 128)
            context_range (int):
                hyper parameter for building dataset. 'context' is subtitle or caption of a clip,
                'range' is the length of the context.
                Examples: 10
        """
        super().__init__()
        # assert len(input_template) == len(output_keys)

        # input_keys, input_suffix = zip(*[t.split('.') for t in input_template])
        # self.input_keys = input_keys
        # self.in2out = dict(zip(input_keys, output_keys))  # in_keys --> out_keys
        # self.out2in = {v: k for k, v in self.in2out.items()}
        # self.input_suffix = input_suffix
        self.pad_key_lens = dict(pad_key_lens) if pad_key_lens is not None else dict()
        # self.missing_mod_default = dict()
        # if missing_mod_default is not None:
        #     for item in missing_mod_default:
        #         self.missing_mod_default[item[0]] = [int(i) for i in item[1:]]

        # +1 when get one missing modality
        # self._call_times = 0
        # self._missing_times = 0
        # self._missing_mod_warning_threshold = 0.6

        # context range
        self.context_range = context_range

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        # input_template = cfg.get("input_template", ['eva_vit_G.npz'])
        # output_keys = cfg.get("output_keys", ['feature_visual'])
        pad_key_lens = cfg.get("pad_key_lens", dict())
        # missing_mod_default = cfg.get("missing_mod_default", dict())
        context_range = cfg.get("context_range", int)

        return cls(
            # input_template=input_template,
            # output_keys=output_keys,
            pad_key_lens=pad_key_lens,
            # missing_mod_default=missing_mod_default,
            context_range=context_range
        )

    # def _get_missing_modality_value(self, mod_name):
    #     if mod_name == 'feature_audio':
    #         if mod_name in self.missing_mod_default:
    #             return torch.zeros(self.missing_mod_default[mod_name])
    #         else:
    #             return torch.zeros(0, 768)
    #     else:
    #         raise ValueError(f"Not able to handle missing modality of {mod_name}")

    # @staticmethod
    # def _get_modality_value(path: Path):
    #     if path.suffix == '.npy':
    #         return torch.tensor(np.load(str(path)))
    #     elif path.suffix == '.npz':
    #         x = np.load(str(path))  # NpzFile
    #         if 'feature' in x.files:
    #             return torch.tensor(x['feature'])
    #         else:
    #             return torch.tensor(x[x.files[0]])
    #     else:
    #         raise ValueError(f"Not able to handle file suffix {path.suffix}")

    @staticmethod
    def _get_vis_feat_value(path: Path):
        x = np.load(str(path))
        return torch.tensor(x['vis_feat'])

    @staticmethod
    def _get_lang_feat_value(path: Path):
        x = np.load(str(path))
        return torch.tensor(x['lang_feat'])

    def __call__(self, fpath, context_feat_path, sub_context, cap_context):
        """
        Args:
            fpath (path):
                (Pseudo) Video feature path to load. --Size is (T, patch_token, dim)--
                Examples: /data/video0.mp4 (real features exist in /data/vit_l/video0.npy)
        Returns:
            torch.tensor: Video feature dict. Size is (T, patch_token, dim).
        """
        # self._call_times += 1

        fpath = Path(fpath)

        output_dict = {}
        context_feat = []
        if fpath.exists():
            output_dict['feature_visual'] = self._get_vis_feat_value(fpath)
        for lang_fpath in context_feat_path:
            lang_fpath = Path(lang_fpath)
            if lang_fpath.exists() and lang_fpath.is_file():
                lang_feat = self._get_lang_feat_value(lang_fpath)[0]
                context_feat.append(lang_feat)
            else:
                lang_feat = torch.zeros(768)
                context_feat.append(lang_feat)
        context_feat = torch.stack(context_feat, dim=0)
        output_dict['feature_language'] = context_feat
        output_dict['subtitle'] = sub_context
        output_dict['caption'] = cap_context
        # for in_name, suffix in zip(self.input_keys, self.input_suffix):
        #     # /data/{vit_l}/{video0}.{npy}
        #     feat_path = fpath.parent / in_name / f"{fpath.stem}.{suffix}"
        #     if feat_path.exists():
        #         # only using numpy for now
        #         output_dict[self.in2out[in_name]] = self._get_modality_value(feat_path)
        #     else:
        #         self._missing_times += 1
        #         output_dict[self.in2out[in_name]] = self._get_missing_modality_value(self.in2out[in_name])

        # if self._call_times > 500 and self._missing_times / self._call_times > self._missing_mod_warning_threshold:
        #     logging.warning(f"Detect TOO MUCH missing modality! missing/all>{self._missing_mod_warning_threshold}")
        #     self._call_times, self._missing_times = 0, 0
        return output_dict

    @staticmethod
    def default_collate(samples_list):
        if type(samples_list[0]) is torch.Tensor:
            return torch.stack(samples_list, dim=0)
        else:
            return samples_list

    @staticmethod
    def pad_collate(samples_list: List[Tensor], max_pad_length):
        """
        Args:
            samples_list (list):  List of [T, dim] or [T, patch, dim]
            max_pad_length (int): the max padded length

        Returns:
            padded_samples (Tensor): [B, max_T, ...]
            pad_mask (BoolTensor): [B, max_T] False is padded
        """
        _pseudo_max_pad_ts = torch.ones([max_pad_length] + list(samples_list[0].shape[1:]))  # [max_T, ...]
        samples_list.append(_pseudo_max_pad_ts)  # add a pseudo max pad tensor to the end

        # 0 is pad; [B, max_T]
        pad_mask = pad_sequence(
            [torch.ones(i.shape[0], dtype=torch.int) for i in samples_list],
            batch_first=True, padding_value=0
        ).bool()
        padded_samples = pad_sequence(samples_list, batch_first=True, padding_value=0.0)
        del samples_list[-1]
        return padded_samples[:-1, :max_pad_length], pad_mask[:-1, :max_pad_length]

    def collate_feature(self, key, samples_list):
        if key in self.pad_key_lens.keys():
            return self.pad_collate(samples_list, self.pad_key_lens[key])
        else:
            return self.default_collate(samples_list)


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)
