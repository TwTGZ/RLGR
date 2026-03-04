#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按用户活跃度分桶对比预训练与后训练模型。
- 同一交互数的样本只在一个桶内
- 各桶样本数尽量平均
- 输出含数据集标识

使用方式:
    python offline_pop_scripts/analyze_activity_buckets.py \
        --data_dir ./data/Toys \
        --pretrained_pred /path/to/pretrained/predictions.json \
        --posttrained_pred /path/to/posttrained/predictions.json \
        --dataset_name Toys \
        --output offline_pop_scripts/activity_bucket_results.txt
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


def create_test_samples(user_seqs: Dict[int, List]) -> List[Tuple[int, int]]:
    """与 TigerDataset test 模式一致"""
    samples = []
    for user_id, item_seq in user_seqs.items():
        if len(item_seq) < 2:
            continue
        samples.append((user_id, len(item_seq)))
    return samples


def build_activity_to_indices(samples: List[Tuple[int, int]]) -> List[Tuple[int, List[int]]]:
    """
    构建 (activity, [样本索引]) 列表，按 activity 升序。
    同一 activity 的样本索引聚在一起。
    """
    act_to_idx = defaultdict(list)
    for idx, (_, activity) in enumerate(samples):
        act_to_idx[activity].append(idx)
    return sorted(act_to_idx.items(), key=lambda x: x[0])


def assign_buckets(
    activity_groups: List[Tuple[int, List[int]]],
    num_buckets: int,
    total_samples: int,
) -> List[Dict]:
    """
    贪心分桶：同一 activity 不拆分，尽量平均各桶样本数。
    当累积样本数达到 target 时即结桶，以尽量生成 num_buckets 个桶。
    若不同 activity 值少于 num_buckets，则桶数 = 不同 activity 的个数。
    """
    target = total_samples / num_buckets
    buckets = []
    current_indices = []
    current_min = current_max = None

    for activity, indices in activity_groups:
        n = len(indices)
        would_be = len(current_indices) + n
        # 达到 target 且还能建新桶时即结桶（阈值 1.05 使更积极分桶）
        need_close = (
            current_indices
            and would_be >= target * 1.05
            and len(buckets) < num_buckets - 1
        )
        if need_close:
            buckets.append({
                "min_act": current_min,
                "max_act": current_max,
                "indices": current_indices.copy(),
            })
            current_indices = []
            current_min = current_max = None

        current_indices.extend(indices)
        current_min = activity if current_min is None else min(current_min, activity)
        current_max = activity if current_max is None else max(current_max, activity)

    if current_indices:
        buckets.append({
            "min_act": current_min,
            "max_act": current_max,
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
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="数据集标识，如 Toys/Beauty/Sports，用于输出")
    parser.add_argument("--num_buckets", type=int, default=4)
    parser.add_argument("--k_list", type=int, nargs="+", default=[10],
                        help="仅保留 Hit@10 和 NDCG@10")
    parser.add_argument("--output", default=None,
                        help="输出文件路径；若不含数据集名则自动追加")
    args = parser.parse_args()

    # 数据集名：优先参数，否则从 data_dir 推断
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    user_seqs = load_user_seqs(args.data_dir)
    samples = create_test_samples(user_seqs)
    n_samples = len(samples)

    with open(args.pretrained_pred, encoding="utf-8") as f:
        pretrained = json.load(f)
    with open(args.posttrained_pred, encoding="utf-8") as f:
        posttrained = json.load(f)

    n_samples = min(n_samples, len(pretrained), len(posttrained))
    samples = samples[:n_samples]

    activity_groups = build_activity_to_indices(samples)
    buckets = assign_buckets(activity_groups, args.num_buckets, n_samples)
    if len(buckets) < args.num_buckets:
        print(
            f"提示: 请求 {args.num_buckets} 个桶，实际得到 {len(buckets)} 个。"
            f"原因: 不同交互数共 {len(activity_groups)} 种，同一交互数不能拆分到多个桶。"
        )

    k_list = args.k_list
    if 10 not in k_list:
        k_list = list(k_list) + [10]
    # 上下对比：每桶两行（预训练 / 后训练），仅输出 Hit@10 / NDCG@10
    cols = ["活跃度桶", "模型", "样本数", "Hit@10", "NDCG@10"]
    w = 12

    n_distinct_act = len(activity_groups)
    lines = [
        f"Dataset: {dataset_name}",
        f"总样本数: {n_samples}",
        f"不同交互数种类: {n_distinct_act} (桶数受此限制)",
        "",
        "".join(c.ljust(w) for c in cols),
        "-" * (len(cols) * w),
    ]

    for b in buckets:
        idx = b["indices"]
        label = f"[{b['min_act']}-{b['max_act']}]"
        n_idx = str(len(idx))
        pre = compute_metrics(pretrained, idx, k_list)
        post = compute_metrics(posttrained, idx, k_list)
        # 预训练行
        lines.append("".join(str(x).ljust(w) for x in [
            label, "预训练", n_idx,
            f"{pre['Hit@10']:.4f}", f"{pre['NDCG@10']:.4f}",
        ]))
        # 后训练行（桶名留空，上下对齐）
        lines.append("".join(str(x).ljust(w) for x in [
            "", "后训练", "",
            f"{post['Hit@10']:.4f}", f"{post['NDCG@10']:.4f}",
        ]))
        lines.append("-" * (len(cols) * w))

    # Overall：同样上下对比
    all_idx = list(range(n_samples))
    pre_a = compute_metrics(pretrained, all_idx, k_list)
    post_a = compute_metrics(posttrained, all_idx, k_list)
    lines.append("".join(str(x).ljust(w) for x in [
        "Overall", "预训练", str(n_samples),
        f"{pre_a['Hit@10']:.4f}", f"{pre_a['NDCG@10']:.4f}",
    ]))
    lines.append("".join(str(x).ljust(w) for x in [
        "", "后训练", "",
        f"{post_a['Hit@10']:.4f}", f"{post_a['NDCG@10']:.4f}",
    ]))

    out = "\n".join(lines)

    output_path = args.output
    if output_path:
        base, ext = os.path.splitext(output_path)
        if dataset_name not in output_path:
            output_path = f"{base}_{dataset_name}{ext}"
        dir_part = os.path.dirname(output_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"结果已保存到: {output_path}")
    print(out)


if __name__ == "__main__":
    main()
