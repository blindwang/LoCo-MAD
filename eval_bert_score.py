from lavis.tasks.video_captioning import video_caption_eval
from pathlib import Path
from bert_score import BERTScorer
import json

# gt_file = "/data/wjy/datasets/MAD/MAD/coco/cap_test_lsmdc_unamed_coco.json"
# gt_file = "/data/wjy/datasets/MAD/MAD/coco/cap_test_lsmdc.json"
gt_file = "/data/wjy/datasets/MAD/MAD/annotation_long_eval.json"
# gt_file = "/data/wjy/datasets/MAD/MAD/annotation_long_unamed_eval.json"
# eval_result_file = f"/data/wjy/workspace/lavis-mm-video-captioning/lavis/output/MAD_Qformer/train_qformer_datav3/20240523110/result/val_epoch0.json"
eval_result_file = "/data/wjy/workspace/lavis-mm-video-captioning/lavis/output/MAD_Qformer/autoad_caption_v4/20240307003/result/val_epoch3.json"
with open(gt_file) as f:
    gt = json.load(f)

with open(eval_result_file) as f:
    eval_result = json.load(f)

candidate, reference = [], []
for res in eval_result:
    img_id = res["image_id"]
    cap = res["caption"]
    candidate.append(cap.lower())
    # reference.append(gt["annotations"][img_id]["caption"].lower())
    reference.append(gt[img_id]["text"].lower())
    # print(cap, gt["annotations"][img_id]["caption"])

# BERTScore calculation
scorer = BERTScorer(model_type='bert-base-uncased', device='cuda')
P, R, F1 = scorer.score(candidate, reference)
# top_k_f1, top_k_f1_idx = F1.topk(100)
top_k_f1_idx = F1.argsort(descending=True)
topk_dict = []
for idx in top_k_f1_idx:
    i = eval_result[idx]["image_id"]
    if gt[i]["movie"] == "1051_Harry_Potter_and_the_goblet_of_fire":
        topk_dict.append({"candidate": eval_result[idx]["caption"], "reference": gt[i]["text"], "score": F1[idx].item(), "idx": i})
with open("/lavis/output/harry_topk_dict_no_stage1.json", "w") as f:
    json.dump(topk_dict, f)
print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
with open(Path(eval_result_file).parent / "bertscore.json", "w") as f:
    json.dump({"P": P.mean().item(), "R": R.mean().item(), "F1": F1.mean().item()}, f)