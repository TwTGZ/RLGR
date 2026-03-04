#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按物品流行度分桶对比预训练与后训练模型。
- 流行度 = 目标物品被多少用户交互过
- 同一流行度的样本只在一个桶内
- 输出 Hit@5, Hit@10, NDCG@5, NDCG@10，上下对比预训练/后训练
"""

import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple


def load_user_seqs(data_dir: str) -> Dict[int, List]:
    path = os.path.join(data_dir, "user2item.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"user2item.pkl 不存在: {path}")
    with open(path, "rb") as f:
        df = pickle.load(f)
    user_seqs = {}
    for _, row in df.iterrows():
        user_id = int(row["UserID"])
        item_seq = list(row["ItemID"])
        user_seqs[user_id] = item_seq
    return user_seqs


def build_item_popularity(user_seqs: Dict[int, List]) -> Dict[int, int]:
    """物品流行度 = 被多少用户交互过（每用户计 1 次）"""
    pop = defaultdict(int)
    for item_seq in user_seqs.values():
        for item in set(item_seq):
            pop[item] += 1
    return dict(pop)


def create_test_samples_with_pop(
    user_seqs: Dict[int, List],
    item_popularity: Dict[int, int],
) -> List[Tuple[int, int]]:
    """
    与 TigerDataset test 一致。(user_id, target_item) -> 样本索引对应 target 的流行度。
    返回: [(sample_idx, popularity), ...]，按样本顺序
    """
    samples_pop = []
    for user_id, item_seq in user_seqs.items():
        if len(item_seq) < 2:
            continue
        target_item = item_seq[-1]
        pop = item_popularity.get(target_item, 0)
        samples_pop.append((target_item, pop))
    return samples_pop


def build_pop_to_indices(
    samples_pop: List[Tuple[int, int]],
) -> List[Tuple[int, List[int]]]:
    """(popularity, [样本索引]) 按流行度升序"""
    pop_to_idx = defaultdict(list)
    for idx, (_, pop) in enumerate(samples_pop):
        pop_to_idx[pop].append(idx)
    return sorted(pop_to_idx.items(), key=lambda x: x[0])


def assign_buckets(
    pop_groups: List[Tuple[int, List[int]]],
    num_buckets: int,
    total_samples: int,
) -> List[Dict]:
    """贪心分桶，同一流行度不拆分"""
    target = total_samples / num_buckets
    buckets = []
    current_indices = []
    current_min = current_max = None

    for pop, indices in pop_groups:
        n = len(indices)
        would_be = len(current_indices) + n
        need_close = (
            current_indices
            and would_be >= target * 1.05
            and len(buckets) < num_buckets - 1
        )
        if need_close:
            buckets.append({
                "min_pop": current_min,
                "max_pop": current_max,
                "indices": current_indices.copy(),
            })
            current_indices = []
            current_min = current_max = None

        current_indices.extend(indices)
        current_min = pop if current_min is None else min(current_min, pop)
        current_max = pop if current_max is None else max(current_max, pop)

    if current_indices:
        buckets.append({
            "min_pop": current_min,
            "max_pop": current_max,
            "indices": current_indices,
        })
    return buckets


def compute_metrics(
    predictions: List[dict],
    indices: List[int],
    k_list: List[int],
) -> Dict[str, float]:
    metrics = {}
    for k in k_list:
        hit_sum = ndcg_sum = 0.0
        for idx in indices:
            sample = predictions[idx]
            label_id = sample["label_id"]
            pred_ids = [p["predicted_id"] for p in sample.get("predictions", [])]
            top_k = pred_ids[:k]
            hit = 1 if label_id in top_k else 0
            hit_sum += hit
            if hit:
                rank = top_k.index(label_id) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
        n = len(indices)
        metrics[f"Hit@{k}"] = hit_sum / n if n else 0.0
        metrics[f"NDCG@{k}"] = ndcg_sum / n if n else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="数据目录，如 ./data/Toys")
    parser.add_argument("--pretrained_pred", required=True, help="预训练 predictions.json")
    parser.add_argument("--posttrained_pred", required=True, help="后训练 predictions.json")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--num_buckets", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_name = args.dataset_name or os.path.basename(os.path.normpath(args.data_dir))

    user_seqs = load_user_seqs(args.data_dir)
    item_popularity = build_item_popularity(user_seqs)
    samples_pop = create_test_samples_with_pop(user_seqs, item_popularity)
    n_samples = len(samples_pop)

    with open(args.pretrained_pred, encoding="utf-8") as f:
        pretrained = json.load(f)
    with open(args.posttrained_pred, encoding="utf-8") as f:
        posttrained = json.load(f)

    n_samples = min(n_samples, len(pretrained), len(posttrained))
    samples_pop = samples_pop[:n_samples]

    pop_groups = build_pop_to_indices(samples_pop)
    buckets = assign_buckets(pop_groups, args.num_buckets, n_samples)
    if len(buckets) < args.num_buckets:
        print(
            f"提示: 请求 {args.num_buckets} 个桶，实际得到 {len(buckets)} 个。"
            f"原因: 不同流行度共 {len(pop_groups)} 种，同一流行度不能拆分。"
        )

    k_list = [5, 10]
    cols = ["流行度桶", "模型", "样本数", "Hit@5", "NDCG@5", "Hit@10", "NDCG@10"]
    w = 12

    lines = [
        f"Dataset: {dataset_name} (按物品流行度分桶)",
        f"总样本数: {n_samples}",
        f"不同流行度种类: {len(pop_groups)}",
        "",
        "".join(c.ljust(w) for c in cols),
        "-" * (len(cols) * w),
    ]

    for b in buckets:
        idx = b["indices"]
        label = f"[{b['min_pop']}-{b['max_pop']}]"
        n_idx = str(len(idx))
        pre = compute_metrics(pretrained, idx, k_list)
        post = compute_metrics(posttrained, idx, k_list)
        lines.append("".join(str(x).ljust(w) for x in [
            label, "预训练", n_idx,
            f"{pre['Hit@5']:.4f}", f"{pre['NDCG@5']:.4f}",
            f"{pre['Hit@10']:.4f}", f"{pre['NDCG@10']:.4f}",
        ]))
        lines.append("".join(str(x).ljust(w) for x in [
            "", "后训练", "",
            f"{post['Hit@5']:.4f}", f"{post['NDCG@5']:.4f}",
            f"{post['Hit@10']:.4f}", f"{post['NDCG@10']:.4f}",
        ]))
        lines.append("-" * (len(cols) * w))

    all_idx = list(range(n_samples))
    pre_a = compute_metrics(pretrained, all_idx, k_list)
    post_a = compute_metrics(posttrained, all_idx, k_list)
    lines.append("".join(str(x).ljust(w) for x in [
        "Overall", "预训练", str(n_samples),
        f"{pre_a['Hit@5']:.4f}", f"{pre_a['NDCG@5']:.4f}",
        f"{pre_a['Hit@10']:.4f}", f"{pre_a['NDCG@10']:.4f}",
    ]))
    lines.append("".join(str(x).ljust(w) for x in [
        "", "后训练", "",
        f"{post_a['Hit@5']:.4f}", f"{post_a['NDCG@5']:.4f}",
        f"{post_a['Hit@10']:.4f}", f"{post_a['NDCG@10']:.4f}",
    ]))

    out = "\n".join(lines)
    if args.output:
        base, ext = os.path.splitext(args.output)
        out_path = f"{base}_pop_{dataset_name}{ext}" if dataset_name not in args.output else args.output
        dir_part = os.path.dirname(out_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"结果已保存到: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
