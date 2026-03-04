#!/bin/bash
# Sports数据集GRPO后训练测试脚本 - 玩具版本
# 用于测试后训练代码是否能正常运行，以及文件结构是否正确
# 2组实验（2个lr），1个epoch

# 切换到项目根目录
cd /home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark

# ========== GPU配置 ==========
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# ========== 测试参数（快速验证用） ==========
model_bsz=512              # 批次大小
model_decay=0.2            # 权重衰减
num_epochs=2               # 只跑1个epoch
eval_epoch=2               # 每1个epoch评估一次
early_stop=5               # 早停耐心值

# GRPO 特有参数
grpo_beta=1                # KL散度系数
grpo_num_gen=4             # 每个样本生成候选数量（测试用较小值）

# ========== 路径配置 ==========
# 预训练模型路径
PRETRAINED_MODEL="/home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark/output/output_sports_multigpu/generation_model"
# output_dir 指向包含 tokenizer_model 的目录
BASE_OUTPUT_DIR="/home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark/output/output_sports_multigpu"
DATA_DIR="./data/Sports"

# ========== 调参范围（测试用，4组） ==========
LR_LIST=(1e-4 5e-4)
# 只测试 match reward
REWARD_LIST=("match" "ndcg")

# ========== 最优结果跟踪 ==========
best_ndcg10=0
best_exp_name=""
best_hit1=0
best_hit5=0
best_hit10=0
best_ndcg1=0
best_ndcg5=0

echo "=============================================="
echo "🧪 测试脚本：验证GRPO后训练框架"
echo "=============================================="
echo "GPU数量: ${NUM_GPUS}"
echo "总批次大小: ${model_bsz} (每卡: $((model_bsz / NUM_GPUS)))"
echo "训练轮数: ${num_epochs}"
echo "学习率范围: ${LR_LIST[*]}"
echo "奖励函数: ${REWARD_LIST[*]}"
echo "GRPO beta: ${grpo_beta}"
echo "GRPO num_generations: ${grpo_num_gen}"
echo "预训练模型: ${PRETRAINED_MODEL}"
echo "=============================================="

# 检查预训练模型是否存在
if [ ! -f "${PRETRAINED_MODEL}/config.json" ]; then
    echo "错误: 预训练模型不存在于 ${PRETRAINED_MODEL}"
    echo "请检查路径"
    exit 1
fi

# 检查tokenizer是否存在
if [ ! -f "${BASE_OUTPUT_DIR}/tokenizer_model/tokenizer.pkl" ]; then
    echo "错误: Tokenizer不存在于 ${BASE_OUTPUT_DIR}/tokenizer_model/"
    echo "请先训练tokenizer或检查路径"
    exit 1
fi

total_exp=$((${#LR_LIST[@]} * ${#REWARD_LIST[@]}))
current_exp=0

# 用于存储所有实验结果的数组
declare -a all_results

for model_lr in "${LR_LIST[@]}"; do
    for reward_type in "${REWARD_LIST[@]}"; do
        current_exp=$((current_exp + 1))
        
        # 构建实验名称和保存路径
        exp_name="lr${model_lr}_${reward_type}"
        posttrain_save_path="${PRETRAINED_MODEL}/posttrain_model/${exp_name}"
        
        echo ""
        echo "=============================================="
        echo "🧪 测试实验 ${current_exp}/${total_exp}: ${exp_name}"
        echo "=============================================="
        echo "学习率: ${model_lr}"
        echo "奖励函数: ${reward_type}"
        echo "模型保存路径: ${posttrain_save_path}"
        echo "=============================================="
        
        # 根据奖励类型设置 reward_func 配置
        if [ "$reward_type" == "match" ]; then
            reward_config="online_rl.trainer.reward_func._target_=genrec.rewards.match_reward.MatchReward"
        else
            # ndcg - GRPOReward with NDCG penalty
            # 注意：使用 + 前缀来添加新的配置键（Hydra语法）
            reward_config="online_rl.trainer.reward_func._target_=genrec.rewards.grpo_reward.GRPOReward +online_rl.trainer.reward_func.use_ndcg=True +online_rl.trainer.reward_func.ndcg_weight=0.5"
        fi
        
        # 检查是否已训练过
        if [ -f "${posttrain_save_path}/config.json" ]; then
            echo "发现已存在的模型，跳过训练..."
        else
            accelerate launch --num_processes=${NUM_GPUS} train_with_online_rl.py \
                skip_tokenizer=True \
                tokenizer.data_text_files="${DATA_DIR}/item2title.pkl" \
                tokenizer.interaction_files="${DATA_DIR}/user2item.pkl" \
                output_dir="${BASE_OUTPUT_DIR}" \
                model.data_interaction_files="${DATA_DIR}/user2item.pkl" \
                model.data_text_files="${DATA_DIR}/item2title.pkl" \
                model.learning_rate=${model_lr} \
                model.weight_decay=${model_decay} \
                model.batch_size=${model_bsz} \
                model.num_epochs=${num_epochs} \
                model.evaluation_epoch=${eval_epoch} \
                model.early_stop_upper_steps=${early_stop} \
                online_rl.trainer.beta=${grpo_beta} \
                online_rl.trainer.num_generations=${grpo_num_gen} \
                ${reward_config} \
                online_rl.pretrained_model="${PRETRAINED_MODEL}" \
                online_rl.save_model_path="${posttrain_save_path}"
        fi
        
        # ========== 解析测试集结果 ==========
        log_file=$(find "${posttrain_save_path}/logs" -name "*.log" -type f 2>/dev/null | head -1)
        
        if [ -z "$log_file" ]; then
            echo "⚠️ 未找到日志文件，跳过结果解析"
            continue
        fi
        
        echo "📋 解析日志: ${log_file}"
        
        # 解析测试集结果
        hit1=$(grep "Hit@1:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*Hit@1: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
        hit5=$(grep "Hit@5:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*Hit@5: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
        hit10=$(grep "Hit@10:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*Hit@10: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
        ndcg1=$(grep "NDCG@1:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*NDCG@1: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
        ndcg5=$(grep "NDCG@5:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*NDCG@5: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
        ndcg10=$(grep "NDCG@10:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*NDCG@10: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
        
        # 检查是否成功解析
        if [ -z "$ndcg10" ]; then
            echo "⚠️ 无法解析测试结果，跳过"
            continue
        fi
        
        echo "📊 测试集结果:"
        echo "   Hit@1:  ${hit1}, Hit@5:  ${hit5}, Hit@10:  ${hit10}"
        echo "   NDCG@1: ${ndcg1}, NDCG@5: ${ndcg5}, NDCG@10: ${ndcg10}"
        
        # 存储结果用于最后汇总
        all_results+=("${exp_name}|${hit1}|${hit5}|${hit10}|${ndcg1}|${ndcg5}|${ndcg10}")
        
        # 更新最优结果
        is_better=$(awk -v a="$ndcg10" -v b="$best_ndcg10" 'BEGIN {print (a > b) ? 1 : 0}')
        if [ "$is_better" -eq 1 ]; then
            best_ndcg10=$ndcg10
            best_exp_name=$exp_name
            best_hit1=$hit1
            best_hit5=$hit5
            best_hit10=$hit10
            best_ndcg1=$ndcg1
            best_ndcg5=$ndcg5
            echo "🏆 新的最优结果！"
        fi
        
        echo "🧪 测试实验 ${exp_name} 完成！"
    done
done

echo ""
echo "=============================================="
echo "📊 所有实验结果汇总"
echo "=============================================="
printf "%-25s %-10s %-10s %-10s %-10s %-10s %-10s\n" "实验" "Hit@1" "Hit@5" "Hit@10" "NDCG@1" "NDCG@5" "NDCG@10"
echo "--------------------------------------------------------------------------------------------------------------"
for result in "${all_results[@]}"; do
    IFS='|' read -r name h1 h5 h10 n1 n5 n10 <<< "$result"
    printf "%-25s %-10s %-10s %-10s %-10s %-10s %-10s\n" "$name" "$h1" "$h5" "$h10" "$n1" "$n5" "$n10"
done

echo ""
echo "=============================================="
echo "🏆 最优实验（基于测试集 NDCG@10）"
echo "=============================================="
if [ -n "$best_exp_name" ]; then
    echo "实验名称: ${best_exp_name}"
    echo "----------------------------------------------"
    echo "📈 测试集六个指标:"
    echo "   Hit@1:   ${best_hit1}"
    echo "   Hit@5:   ${best_hit5}"
    echo "   Hit@10:  ${best_hit10}"
    echo "   NDCG@1:  ${best_ndcg1}"
    echo "   NDCG@5:  ${best_ndcg5}"
    echo "   NDCG@10: ${best_ndcg10} (选择依据)"
    echo "----------------------------------------------"
    echo "模型路径: ${PRETRAINED_MODEL}/posttrain_model/${best_exp_name}"
else
    echo "⚠️ 没有成功完成的实验"
fi
echo "=============================================="

# 展示实际生成的目录
echo ""
echo "实际生成的目录："
ls -la "${PRETRAINED_MODEL}/posttrain_model/" 2>/dev/null || echo "目录不存在"
