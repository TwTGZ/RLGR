#!/bin/bash
# Beauty 数据集：按用户活跃度分桶对比预训练 vs 后训练模型

cd /home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark

# ========== Beauty 数据集 ==========
DATA_DIR="./data/Beauty"
DATASET_NAME="Beauty"
PRETRAINED_PRED="/home/notebook/data/group/RLGR/output_beauty_ut/generation_model/lr3e-3_wd0.3/predictions.json"
POSTTRAINED_PRED="/home/notebook/data/group/RLGR/output_beauty_ut/generation_model/lr3e-3_wd0.3/offline_rere_final/lr1e-4_beta1/predictions.json"
OUTPUT="offline_pop_scripts/activity_bucket_results.txt"

python offline_pop_scripts/analyze_activity_buckets.py \
    --data_dir "${DATA_DIR}" \
    --pretrained_pred "${PRETRAINED_PRED}" \
    --posttrained_pred "${POSTTRAINED_PRED}" \
    --dataset_name "${DATASET_NAME}" \
    --num_buckets 4 \
    --k_list 10 \
    --output "${OUTPUT}"
