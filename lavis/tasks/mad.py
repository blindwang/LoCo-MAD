"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

import wandb

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.video_captioning import video_caption_eval


@registry.register_task("mad")
class MADTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, gt_files, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.gt_files = gt_files

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        gt_files = run_cfg.gt_files

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            gt_files=gt_files,
            report_metric=report_metric,
        )

    def train_step(self, model, samples):
        output = model(samples)
        if hasattr(model, 'loss_config') and 'DAL' in model.loss_config:
            wandb.log({'loss_dal': output['loss_dal']})
        return output["loss"], output["select_loss"]

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": img_id})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                gt_file=self.gt_files[split_name], eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, gt_file, eval_result_file, split_name):
        gt_file = Path(gt_file)
        if not gt_file.is_absolute():
            gt_file = Path(registry.get_path('cache_root')) / gt_file

        coco_val = video_caption_eval(gt_file, eval_result_file)

        agg_metrics = coco_val.eval["CIDEr"] # + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        # with open(os.path.join(registry.get_path("output_dir"), "evaluate_detail.txt"), "w+") as f:
        #     json.dump(coco_val.imgToEval, f)

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        wandb.log(data=coco_res)
        # wandb.run.log(data=coco_res)

        return coco_res

