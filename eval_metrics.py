from lavis.tasks.video_captioning import video_caption_eval
from pathlib import Path
import json
import os
gt_file = "/data/wjy/datasets/LSMDC/MLLM_res/anno_coco.json"
mods = ["context", "pre", "recurrent"]
for mod in mods[:1]:
    eval_result_file = f"/data/wjy/datasets/LSMDC/MLLM_res/{mod}_res.json"
    gt_file = Path(gt_file)

    coco_val = video_caption_eval(gt_file, eval_result_file)

    agg_metrics = coco_val.eval["CIDEr"] # + coco_val.eval["Bleu_4"]
    log_stats = {f"{mod}_res_infer": {k: v for k, v in coco_val.eval.items()}}

    output_dir = "lavis/output"
    with open(os.path.join(output_dir, "evaluate_videollava.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")
    # with open(os.path.join(registry.get_path("output_dir"), "evaluate_detail.txt"), "w+") as f:
    #     json.dump(coco_val.imgToEval, f)

    coco_res = {k: v for k, v in coco_val.eval.items()}
    coco_res["agg_metrics"] = agg_metrics
    print(coco_res)