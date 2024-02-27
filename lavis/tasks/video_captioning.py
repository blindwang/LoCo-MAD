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


@registry.register_task("video_captioning")
class VideoCaptionTask(BaseTask):
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
        return output["loss"]

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
        with open(os.path.join(registry.get_path("output_dir"), "evaluate_detail.txt"), "w+") as f:
            json.dump(coco_val.imgToEval, f)

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        wandb.log(data=coco_res)
        # wandb.run.log(data=coco_res)

        return coco_res


@registry.register_task("ch_video_captioning")
class ChVideoCaptionTask(VideoCaptionTask):
    @main_process
    def _report_metrics(self, gt_file, eval_result_file, split_name):
        gt_file = Path(gt_file)
        if not gt_file.is_absolute():
            gt_file = Path(registry.get_path('cache_root')) / gt_file

        coco_val = video_caption_chinese_eval(gt_file, eval_result_file)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
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

@registry.register_task("flop")
class FlopTask(VideoCaptionTask):

    def train_step(self, model, samples):
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, samples)
        print(flops.by_module_and_operator())
        print(flops.total())
        assert 1 == 0
        # loss = model(samples)["loss"]
        # return loss


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def video_caption_eval(gt_file, results_file):
    # create coco object and coco_result object
    coco = COCO(gt_file)
    coco_result = coco.loadRes(results_file)
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()
    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def video_caption_chinese_eval(gt_file, results_file):
    import language_evaluation.coco_caption_py3.pycocoevalcap as eval_tools
    import jieba
    raw_gts = json.load(open(gt_file))['annotations']
    gts = defaultdict(list)
    for i in raw_gts:
        gts[i['image_id']].append(i['caption'])

    raw_results = json.load(open(results_file))
    results = defaultdict(list)
    for i in raw_results:
        results[i['image_id']].append(" ".join(jieba.cut(i['caption'])))

    class ChEvalResults:
        def __init__(self):
            self.eval = {}

    res = ChEvalResults()

    # ciders = eval_tools.compute_ciders(gts, results)
    # res.eval['CIDEr'] = ciders[0]
    # bleus = eval_tools.compute_bleus(gts, results)
    # bleu_sum = np.zeros(4)
    # for item in bleus:
    #     bleu_sum += list(item.values())[0]
    # bleu_sum /= len(gts)
    # res.eval['Bleu_1'], res.eval['Bleu_2'], res.eval['Bleu_3'], res.eval['Bleu_4'] = bleu_sum

    all_score, all_scores = eval_tools.compute_scores(gts, results)

    import pickle
    with open(os.path.join(registry.get_path("output_dir"), "evaluate_detail.pkl"), "wb+") as f:
        pickle.dump(all_scores, f)

    metrics = ('Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')
    for i, v in enumerate(all_score.values()):
        if type(v) is list:
            res.eval['Bleu_1'], res.eval['Bleu_2'], res.eval['Bleu_3'], res.eval['Bleu_4'] = v
        else:
            res.eval[metrics[i]] = v

    # print output evaluation scores
    for metric, score in res.eval.items():
        print(f"{metric}: {score:.3f}")

    return res

