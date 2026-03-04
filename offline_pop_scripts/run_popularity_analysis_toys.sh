#!/bin/bash
# Toys 数据集：按物品流行度分桶对比预训练 vs 后训练
# 输出 Hit@5, Hit@10, NDCG@5, NDCG@10

cd /home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark

DATA_DIR="./data/Toys"
DATASET_NAME="Toys"
PRETRAINED_PRED="/home/notebook/data/group/RLGR/output_toys_ut/generation_model/lr4e-3_wd0.7/predictions.json"
POSTTRAINED_PRED="/home/notebook/data/group/RLGR/output_toys_ut/generation_model/lr4e-3_wd0.7/offline_rere_final/lr1e-4_beta1/predictions.json"
OUTPUT="offline_pop_scripts/popularity_bucket_results.txt"

python offline_pop_scripts/analyze_popularity_buckets.py \
    --data_dir "${DATA_DIR}" \
    --pretrained_pred "${PRETRAINED_PRED}" \
    --posttrained_pred "${POSTTRAINED_PRED}" \
    --dataset_name "${DATASET_NAME}" \
    --num_buckets 4 \
    --output "${OUTPUT}"
