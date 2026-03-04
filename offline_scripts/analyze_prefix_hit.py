#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 predictions.json 中 Top-1 预测的前缀命中率
直接打印到终端，不保存到文件
"""

import json
import argparse
import os


def analyze_prefix_hit_rate(predictions_path: str, max_prefix_len: int = 4):
    if os.path.isdir(predictions_path):
        predictions_path = os.path.join(predictions_path, "predictions.json")

    if not os.path.exists(predictions_path):
        print(f"错误: 文件不存在 {predictions_path}")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    if total == 0:
        print("没有样本")
        return

    hit_counts = [0] * (max_prefix_len + 1)

    for sample in results:
        label_tokens = sample.get("label_tokens", [])
        preds = sample.get("predictions", [])
        if not preds:
            continue

        top1_tokens = preds[0].get("predicted_tokens", [])
        if not label_tokens or not top1_tokens:
            continue

        matched_len = 0
        for k in range(1, min(max_prefix_len, len(label_tokens), len(top1_tokens)) + 1):
            if label_tokens[:k] == top1_tokens[:k]:
                matched_len = k
            else:
                break

        hit_counts[matched_len] += 1

    print("=" * 50)
    print(f"Top-1 前缀命中统计 (总样本数: {total})")
    print(f"数据: {predictions_path}")
    print("=" * 50)
    print(f"{'前缀长度':<12} {'命中样本数':<12} {'命中率':<12} {'累计命中率(>=)'}")
    print("-" * 50)

    cum = 0
    for k in range(1, max_prefix_len + 1):
        cum += hit_counts[k]
        rate = hit_counts[k] / total
        cum_rate = cum / total
        print(f"前 {k} 个 token   {hit_counts[k]:<12} {rate:.4f}       {cum_rate:.4f}")

    full_match = hit_counts[max_prefix_len]
    print("-" * 50)
    print(f"完整命中(4/4) {full_match:<12} {full_match/total:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 Top-1 前缀命中率（仅打印，不保存）")
    parser.add_argument("path", type=str, nargs="?", default="/home/notebook/data/group/RLGR/output_toys_ut/generation_model/lr4e-3_wd0.7/predictions.json", help="predictions.json 路径或模型目录")
    parser.add_argument("--max_prefix", type=int, default=4, help="最大前缀长度")
    args = parser.parse_args()

    analyze_prefix_hit_rate(args.path, args.max_prefix)