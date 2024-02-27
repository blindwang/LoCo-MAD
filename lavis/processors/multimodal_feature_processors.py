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
from transformers import AutoProcessor, BatchFeature, ASTFeatureExtractor

from typing import List
from torch import Tensor
from pathlib import Path
import re

MAX_INT = registry.get("MAX_INT")


@registry.register_processor("mmvct_video")
class MMVCTVideoProcessor(BaseProcessor):
    def __init__(
            self,
            ftype="npz",
            keys=None,
            padded_keys=None,
            max_pad_length=16
    ):
        super().__init__()

        if keys is None:
            self.keys = ['feature_visual']
        else:
            self.keys = keys
        assert ftype == 'npz'
        self.ftype = ftype

        self.padded_keys = [] if padded_keys is None else padded_keys
        self.max_pad_length = max_pad_length

    def __call__(self, fpath):
        """
        Args:
            fpath (path): Video feature path to load. Size is (T, patch_token, dim)
        Returns:
            torch.tensor: Video feature dict. Size is (T, patch_token, dim).
        """
        npzfile = np.load(fpath, allow_pickle=True)
        output_dict = {}
        for k, v in npzfile.items():
            if k in self.keys and len(v.shape) != 0:
                output_dict[k] = torch.tensor(v)
        # deal with missing modalities
        missing = tuple(set(self.keys) - set(output_dict.keys()))
        for mod in missing:
            if mod == 'feature_audio':
                output_dict['feature_audio'] = torch.ones(0, 768)
            else:
                raise ValueError("Not able to handle missing modality")

        return output_dict

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        ftype = cfg.get("ftype", "npz")
        keys = cfg.get("keys", ['feature_visual'])
        padded_keys = cfg.get("padded_keys", None)
        max_pad_length = cfg.get("max_pad_length", 16)

        return cls(
            ftype=ftype,
            keys=keys,
            padded_keys=padded_keys,
            max_pad_length=max_pad_length
        )

    def default_collate(self, samples_list):
        if type(samples_list[0]) is torch.Tensor:
            return torch.stack(samples_list, dim=0)
        else:
            return samples_list

    def pad_collate(self, samples_list: List[Tensor]):
        """
        Args:
            samples_list (list):  List of [T, dim] or [T, patch, dim]

        Returns:
            padded_samples (Tensor): [B, max_T, ...]
            pad_mask (BoolTensor): [B, max_T] False is padded
        """
        _pseudo_max_pad_ts = torch.ones([self.max_pad_length] + list(samples_list[0].shape[1:]))  # [max_T, ...]
        samples_list.append(_pseudo_max_pad_ts)
        # 0 is pad; [B, max_T]
        pad_mask = pad_sequence(
            [torch.ones(i.shape[0], dtype=torch.int) for i in samples_list],
            batch_first=True, padding_value=0
        ).bool()
        padded_samples = pad_sequence(samples_list, batch_first=True, padding_value=0.0)
        del samples_list[-1]
        return padded_samples[:-1, :self.max_pad_length], pad_mask[:-1, :self.max_pad_length]

    def collate_feature(self, key, samples_list):
        if key in self.padded_keys:
            return self.pad_collate(samples_list)
        else:
            return self.default_collate(samples_list)


@registry.register_processor("val_qformer_feature")
class VALQformerFeatureProcessor(BaseProcessor):
    def __init__(
            self,
            input_template,
            output_keys,
            pad_key_lens=None,
            missing_mod_default=None,
            drop_key_ratios=None
    ):
        """
        Args:
            input_template (list):
                List of template string `folder_name.suffix`
                Examples: eva_vit_G.npz
            output_keys (list):
                List of output keys in string. The order corresponds to `input_template`
            pad_key_lens (list):
                List of key-length tuple. `Key` is the output feature key which needs padded,
                `length` is the pad length.
                 Examples: ('feature_visual', 12), ('feature_audio', 8)
            missing_mod_default (list):
                List of key-dims tuple. `Key` is the output feature key which may missing,
                `dims` are the default dimension.
                Examples: ('feature_audio', 0, 128)
        """
        super().__init__()
        assert len(input_template) == len(output_keys)

        input_keys, input_suffix = zip(*[t.split('.') for t in input_template])
        self.input_keys = input_keys
        self.in2out = dict(zip(input_keys, output_keys))  # in_keys --> out_keys
        self.out2in = {v: k for k, v in self.in2out.items()}
        self.input_suffix = input_suffix
        self.pad_key_lens = dict(pad_key_lens) if pad_key_lens is not None else dict()
        self.missing_mod_default = dict()
        if missing_mod_default is not None:
            for item in missing_mod_default:
                self.missing_mod_default[item[0]] = [int(i) for i in item[1:]]
        self.drop_key_ratios = dict(drop_key_ratios) if drop_key_ratios is not None else dict()
        self.drop_key_ratios = {k: float(v) for k, v in self.drop_key_ratios.items()}

        # +1 when get one missing modality
        self._call_times = 0
        self._missing_times = 0
        self._missing_mod_warning_threshold = 0.6

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        input_template = cfg.get("input_template", ['eva_vit_G.npz'])
        output_keys = cfg.get("output_keys", ['feature_visual'])
        pad_key_lens = cfg.get("pad_key_lens", dict())
        drop_key_ratios = cfg.get("drop_key_ratios", dict())
        missing_mod_default = cfg.get("missing_mod_default", dict())

        return cls(
            input_template=input_template,
            output_keys=output_keys,
            pad_key_lens=pad_key_lens,
            drop_key_ratios=drop_key_ratios,
            missing_mod_default=missing_mod_default,
        )

    def _get_missing_modality_value(self, mod_name):
        if mod_name == 'feature_audio':
            if mod_name in self.missing_mod_default:
                return torch.zeros(self.missing_mod_default[mod_name])
            else:
                return torch.zeros(0, 768)
        else:
            raise ValueError(f"Not able to handle missing modality of {mod_name}")

    @staticmethod
    def _get_modality_value(path: Path):
        if path.suffix == '.npy':
            return torch.tensor(np.load(str(path)))
        elif path.suffix == '.npz':
            x = np.load(str(path))  # NpzFile
            if 'feature' in x.files:
                return torch.tensor(x['feature'])
            else:
                return torch.tensor(x[x.files[0]])
        else:
            raise ValueError(f"Not able to handle file suffix {path.suffix}")

    def __call__(self, fpath):
        """
        Args:
            fpath (path):
                (Pseudo) Video feature path to load. --Size is (T, patch_token, dim)--
                Examples: /data/video0.mp4 (real features exist in /data/vit_l/video0.npy)
        Returns:
            torch.tensor: Video feature dict. Size is (T, patch_token, dim).
        """
        self._call_times += 1

        fpath = Path(fpath)
        output_dict = {}
        for in_name, suffix in zip(self.input_keys, self.input_suffix):
            # /data/{vit_l}/{video0}.{npy}
            feat_path = fpath.parent / in_name / f"{fpath.stem}.{suffix}"
            if feat_path.exists():
                # only using numpy for now
                output_dict[self.in2out[in_name]] = self._get_modality_value(feat_path)
            else:
                self._missing_times += 1
                output_dict[self.in2out[in_name]] = self._get_missing_modality_value(self.in2out[in_name])

        if self._call_times > 500 and self._missing_times / self._call_times > self._missing_mod_warning_threshold:
            logging.warning(f"Detect TOO MUCH missing modality! missing/all>{self._missing_mod_warning_threshold}")
            self._call_times, self._missing_times = 0, 0
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

    @staticmethod
    def drop_feature(feature: Tensor, drop_ratio: float):
        feature = feature[:, :, 1:]
        b, t, k, c = feature.shape
        if drop_ratio == 0.0:
            select_idx = torch.arange(0, k, 1)[None, None, :].repeat(b, t, 1)
        else:
            # [B, T, K, C] -> [B, T, (1-drop_ratio)K, C]
            select_idx = torch.multinomial(
                torch.ones(b*t, k), int((1 - drop_ratio) * k)
            ).view(b, t, -1)  # [B, T, (1-drop_ratio)K]
            # select_idx = select_idx.unsqueeze(-1).expand(-1, -1, -1, feature.shape[-1])
            feature = torch.gather(feature, dim=2, index=select_idx.unsqueeze(-1).expand(-1, -1, -1, feature.shape[-1]))
        return feature, select_idx

    def collate_feature(self, key, samples_list):
        if key in self.pad_key_lens:
            out = self.pad_collate(samples_list, self.pad_key_lens[key])
        else:
            out = self.default_collate(samples_list)
        if key in self.drop_key_ratios:
            out = self.drop_feature(out, self.drop_key_ratios[key])
        return out

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


class VALQformerVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT, audio=False,
                 audio_sample_rate=16000, audio_window_size=2.0, audio_overlap=0.5, audio_max_len=8,
                 ffmpeg_path='ffmpeg'):
        super().__init__()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms
        self.audio = audio
        if audio:
            self.sample_rate = audio_sample_rate
            self.window_size = audio_window_size
            self.overlap = audio_overlap
            self.audio_max_len = audio_max_len
            self.ffmpeg_path = ffmpeg_path
            # self.audio_proc = AutoProcessor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
            self.audio_proc = ASTFeatureExtractor(padding_side="right", padding_value=0.0, return_attention_mask=False,
                                                  sampling_rate=16000, do_normalize=True)

    @staticmethod
    def extract_audio_from_mp4(video_path, ffmpeg_path='ffmpeg'):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(video_path)
            wav_path = Path(tmp_dir, f"{video_path.stem}.wav")

            mp4_to_wav_cmd = f"{ffmpeg_path} -hide_banner -loglevel panic -y -i {video_path} {wav_path}"
            subprocess.call(mp4_to_wav_cmd.split())
            if not wav_path.exists():
                raise FileNotFoundError

            with soundfile.SoundFile(wav_path) as f:
                audio_raw = f.read(dtype=np.float32)
                if len(audio_raw.shape) > 1:
                    audio_raw = audio_raw.mean(axis=1)
                sample_rate_raw = f.samplerate

            return audio_raw, sample_rate_raw

    def extract_audio_segments(self, vp):
        try:
            audio_raw, sample_rate_raw = self.extract_audio_from_mp4(vp, ffmpeg_path=self.ffmpeg_path)
        except FileNotFoundError:
            return None
        audio_resample = resampy.resample(audio_raw, sample_rate_raw, self.sample_rate)

        ws = int(self.window_size * self.sample_rate)
        step = int((self.window_size - self.overlap) * self.sample_rate)
        start = 0
        segments = []
        while True:
            seg = audio_resample[start: start + ws]
            if len(seg) != ws:
                break
            # print(f"{start} -> {start + ws}")
            segments.append(seg)
            start += step
        return segments

    def collate_feature(self, key, samples_list):
        if key == 'video':
            return torch.stack(samples_list, dim=0)
        elif key == 'audio':
            # Input: List of [T, c1, c2]
            _pseudo_max_pad_ts = torch.ones([self.audio_max_len] + list(samples_list[0].shape[1:]))  # [max_T, ...]
            samples_list.append(_pseudo_max_pad_ts)  # add a pseudo max pad tensor to the end
            # [B+1, max_T]
            pad_mask = pad_sequence(
                [torch.ones(i.shape[0], dtype=torch.int) for i in samples_list],
                batch_first=True, padding_value=0
            ).bool()
            # [B+1, max_T, c1, c2]
            padded_samples = pad_sequence(samples_list, batch_first=True, padding_value=0.0)
            del samples_list[-1]
            return padded_samples[:-1, :self.audio_max_len], pad_mask[:-1, :self.audio_max_len]
        else:
            # raise ValueError(f"Cannot handle feature of {key}")
            return samples_list


@registry.register_processor("val_qformer_video_train")
class VALQformerVideoTrainProcessor(VALQformerVideoBaseProcessor):
    def __init__(
            self,
            image_size=384,
            mean=None,
            std=None,
            min_scale=0.5,
            max_scale=1.0,
            n_frms=MAX_INT,
            audio=False,
            audio_sample_rate=16000,
            audio_window_size=2.0,
            audio_overlap=0.5,
            ffmpeg_path='ffmpeg'
    ):
        super().__init__(
            mean=mean,
            std=std,
            n_frms=n_frms,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            audio_window_size=audio_window_size,
            audio_overlap=audio_overlap,
            ffmpeg_path=ffmpeg_path
        )

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                transforms_video.RandomHorizontalFlipVideo(),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                VideoRandomAugment(
                    2,
                    5,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        vpath = Path(vpath)
        if vpath.suffix != '.mp4':
            vpath = str(vpath.parent / f"{vpath.stem}.mp4")
        output_dict = {}
        clip = load_video(
            video_path=str(vpath),
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )
        output_dict['video'] = self.transform(clip)

        if self.audio is True:
            segments = self.extract_audio_segments(vp=vpath)  # may be None
            if segments is None or len(segments) < 1:
                audio_inputs = torch.zeros((0, 1024, 128), dtype=torch.float32)
            else:
                # len, 1024, 128
                audio_inputs = self.audio_proc(
                    segments, sampling_rate=self.sample_rate, return_tensors="pt"
                )['input_values']
            output_dict['audio'] = audio_inputs

        return output_dict

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)

        audio = cfg.get("audio", False)
        audio_sample_rate = cfg.get("audio_sample_rate", 16000)
        audio_window_size = cfg.get("audio_window_size", 2.0)
        audio_overlap = cfg.get("audio_overlap", 0.5)
        ffmpeg_path = cfg.get("ffmpeg_path", 'ffmpeg')

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
            audio=audio,
            audio_sample_rate=audio_sample_rate, audio_window_size=audio_window_size,
            audio_overlap=audio_overlap, ffmpeg_path=ffmpeg_path
        )


@registry.register_processor("val_qformer_video_eval")
class VALQformerVideoEvalProcessor(VALQformerVideoBaseProcessor):
    def __init__(
            self,
            image_size=256,
            mean=None,
            std=None,
            n_frms=MAX_INT,
            audio=False,
            audio_sample_rate=16000,
            audio_window_size=2.0,
            audio_overlap=0.5,
            ffmpeg_path='ffmpeg'
    ):
        super().__init__(
            mean=mean,
            std=std,
            n_frms=n_frms,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            audio_window_size=audio_window_size,
            audio_overlap=audio_overlap,
            ffmpeg_path=ffmpeg_path
        )

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        vpath = Path(vpath)
        if vpath.suffix != '.mp4':
            vpath = str(vpath.parent / f"{vpath.stem}.mp4")
        output_dict = {}
        clip = load_video(
            video_path=str(vpath),
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
        )
        clip = self.transform(clip)
        if clip.shape[1] < self.n_frms:
            output_dict['video'] = torch.cat([
                clip,
                clip.new_zeros(3, self.n_frms - clip.shape[1], self.image_size, self.image_size)
            ], dim=1)
            print(f"Padding {vpath.name} from {clip.shape[1]} to {self.n_frms}")
        else:
            output_dict['video'] = clip

        if self.audio is True:
            segments = self.extract_audio_segments(vp=vpath)  # may be None
            print(len(segments))
            if segments is None or len(segments) < 1:
                audio_inputs = torch.zeros((0, 1024, 128), dtype=torch.float32)
            else:
                # len, 1024, 128
                audio_inputs = self.audio_proc(
                    segments, sampling_rate=self.sample_rate, return_tensors="pt"
                )['input_values']
            output_dict['audio'] = audio_inputs

        return output_dict

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)
        audio = cfg.get("audio", False)
        audio_sample_rate = cfg.get("audio_sample_rate", 16000)
        audio_window_size = cfg.get("audio_window_size", 2.0)
        audio_overlap = cfg.get("audio_overlap", 0.5)
        ffmpeg_path = cfg.get("ffmpeg_path", 'ffmpeg')

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms, audio=audio,
                   audio_sample_rate=audio_sample_rate, audio_window_size=audio_window_size,
                   audio_overlap=audio_overlap, ffmpeg_path=ffmpeg_path)


@registry.register_processor("val_qformer_caption")
class VALQformerCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50, no_period=False):
        self.prompt = prompt
        self.max_words = max_words
        self.no_period = no_period

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)
        no_period = cfg.get("no_period", False)

        return cls(prompt=prompt, max_words=max_words, no_period=no_period)

    def pre_caption(self, caption):
        if self.no_period:
            caption = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                caption.lower(),
            )
        else:
            caption = re.sub(
                r"([!\"()*#:;~])",
                " ",
                caption.lower(),
            )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("val_qformer_chinese_caption")
class VALQformerChCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50, no_period=False):
        self.prompt = prompt
        self.max_words = max_words
        self.no_period = no_period

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)
        no_period = cfg.get("no_period", False)

        return cls(prompt=prompt, max_words=max_words, no_period=no_period)

    def pre_caption(self, caption):
        if self.no_period:
            caption = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                caption.lower(),
            )
        else:
            caption = re.sub(
                r"([!\"()*#:;~])",
                " ",
                caption.lower(),
            )
        caption = re.sub(r"\s{2,}", " ", caption)
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        # caption_words = caption.split(" ")
        # if len(caption_words) > self.max_words:
        #     caption = " ".join(caption_words[: self.max_words])
        if len(caption) > self.max_words:
            caption = caption[: self.max_words]

        return caption
