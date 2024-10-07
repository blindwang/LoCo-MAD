import json

import torch
from bert_score import BERTScorer  # from https://github.com/Tiiiger/bert_score
from tqdm import tqdm
from collections import defaultdict

bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device="cuda")


def recall_within_neighbours(sentences_gt, sentences_gen, topk=(1, 5), N=16):
    """compute R@k/N as described in AutoAD-II (https://www.robots.ox.ac.uk/~vgg/publications/2023/Han23a/han23a.pdf)
    This metric compares a (long) list of sentences with another list of sentences.
    It uses BERTScore (https://github.com/Tiiiger/bert_score) to compute sentence-sentence similarity,
    but uses the relative BERTScore values to get a recall, for robustness.
    """
    # get sentence-sentence BertScore
    ss_score = []
    for sent in sentences_gen:
        ss_score.append(bert_scorer.score(sentences_gt, [sent] * len(sentences_gt))[-1])
    ss_score = torch.stack(ss_score, dim=0)

    window = N
    topk_output = []
    # stride: window // 2
    for i in range(0, ss_score.shape[0] - window + 1, window // 2):
        topk_output.append(
            calc_topk_accuracy(ss_score[i:i + window, i:i + window], torch.arange(window).to(ss_score.device),
                               topk=topk))

    topk_avg = torch.stack(topk_output, 0).mean(0).tolist()
    for k, res in zip(topk, topk_avg):
        print(f"Recall@{k}/{N}: {res:.3f}")
    return topk_avg


def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return torch.stack(res)


if __name__ == '__main__':
    # Example
    # in practice, we put all the ADs of a movie in sentences_gt and sentences_gen
    gt_file = "/data/wjy/datasets/MAD/MAD/coco/cap_test_lsmdc.json"
    # eval_result_file = f"/data/wjy/workspace/lavis-mm-video-captioning/lavis/output/MAD_Qformer/train_qformer_datav3/20240523110/result/val_epoch0.json"
    eval_result_file = "/data/wjy/workspace/lavis-mm-video-captioning/lavis/output/MAD_Qformer/autoad_caption_recurrent_datav3/20240610164/result/val_epoch0.json"

    data_w_movie_info = "/data/wjy/datasets/MAD/MAD/annotation_long_eval.json"
    with open(gt_file) as f:
        gt = json.load(f)

    with open(eval_result_file) as f:
        eval_result = json.load(f)

    with open(data_w_movie_info) as f:
        data = json.load(f)

    # sort by image id
    gen_data = sorted(eval_result, key=lambda x: x["image_id"])
    # collate according to movie id
    collated_data = defaultdict(list)
    for item in data:
        if item["movie"] not in collated_data:
            collated_data[item["movie"]] = [[], []]
        collated_data[item["movie"]][0].append(item["text"])
        collated_data[item["movie"]][1].append(gen_data[item["index"]]["caption"])

    topk_all = []
    for movie_id in tqdm(collated_data):
        sentences_gt = collated_data[movie_id][0][:100]
        sentences_gen = collated_data[movie_id][1][:100]
        result = recall_within_neighbours(sentences_gt, sentences_gen, topk=(5,), N=16)
        topk_all.append(result)
    # print(sum(topk_all) / len(topk_all))
    # Should get
    # Recall@1/4: 0.667
    # Recall@3/4: 0.917