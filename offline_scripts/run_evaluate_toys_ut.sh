#!/bin/bash
# Toys 数据集模型评估脚本（使用 user_tokens）
# 用于测试已训练但未完成测试评估的模型
# 使用单 GPU 运行以避免多卡同步问题

cd /home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark

# ========== 配置 ==========
# 使用单 GPU
export CUDA_VISIBLE_DEVICES=0

# 路径配置
BASE_OUTPUT_DIR="/home/notebook/data/group/RLGR/output_sports_ut"
TOKENIZER_PATH="${BASE_OUTPUT_DIR}/tokenizer_model/tokenizer.pkl"
DATA_DIR="./data/Sports"

# 模型配置
USE_USER_TOKENS="--use_user_tokens"  # 如果不使用 user_tokens，将此行注释掉或改为空字符串
BATCH_SIZE=256
MAX_SEQ_LEN=20
NUM_BEAMS=10
MAX_GEN_LENGTH=5

# ========== 要测试的模型列表 ==========
# 可以添加多个模型路径
MODEL_PATHS=(
    # Off-policy GRPO with Prefix reward
    "${BASE_OUTPUT_DIR}/generation_model/lr3e-3_wd0.4/offline_rere_final/lr1e-4_beta1"

    # "${BASE_OUTPUT_DIR}/generation_model/lr4e-3_wd0.7/offline_rere_wgt/lr3e-3/checkpoint-26350"

    # Off-policy GRPO with NDCG reward  
    # "${BASE_OUTPUT_DIR}/generation_model/lr3e-3_wd0.3/offline_rere/lr2e-3/checkpoint-64100"
)

echo "=============================================="
echo "Toys 数据集模型评估（单 GPU 模式）"
echo "=============================================="
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "数据目录: ${DATA_DIR}"
echo "使用 user_tokens: ${USE_USER_TOKENS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num beams: ${NUM_BEAMS}"
echo "=============================================="
echo ""

# 检查 tokenizer 是否存在
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "错误: Tokenizer 不存在于 ${TOKENIZER_PATH}"
    exit 1
fi

# 遍历所有模型进行测试
for model_path in "${MODEL_PATHS[@]}"; do
    echo ""
    echo "=============================================="
    echo "测试模型: ${model_path}"
    echo "=============================================="
    
    # 检查模型是否存在
    if [ ! -f "${model_path}/config.json" ]; then
        echo "⚠️ 模型不存在或未完成训练，跳过: ${model_path}"
        continue
    fi
    
    # 设置日志目录（保存到模型目录下）
    LOG_DIR="${model_path}/eval_logs"
    
    # 运行评估
    python offline_scripts/evaluate_model.py \
        --model_path "${model_path}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --data_dir "${DATA_DIR}" \
        ${USE_USER_TOKENS} \
        --batch_size ${BATCH_SIZE} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --num_beams ${NUM_BEAMS} \
        --max_gen_length ${MAX_GEN_LENGTH} \
        --k_list 1 5 10 \
        --log_dir "${LOG_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "✅ 模型评估完成: ${model_path}"
    else
        echo "❌ 模型评估失败: ${model_path}"
    fi
done

echo ""
echo "=============================================="
echo "所有模型评估完成"
echo "=============================================="
