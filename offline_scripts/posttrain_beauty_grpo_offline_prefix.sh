#!/bin/bash
# Beauty数据集GRPO Off-Policy后训练脚本 - Prefix奖励版本（使用user_tokens）
# 基于预训练模型进行后训练，调节学习率
# 每个实验保存到独立的文件夹，日志也独立
# 训练结束后汇总所有结果并输出最优参数

# 切换到项目根目录
cd /home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark

# ========== GPU配置 ==========
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# ========== 固定参数 ==========
model_bsz=1024             # 总批次大小
model_decay=0.2            # 权重衰减（使用默认）
num_epochs=500             # 训练轮数（有早停）
eval_epoch=5               # 每5个epoch评估一次
eval_start_epoch=25        # 前25个epoch不评估（从第30个epoch开始评估）
early_stop=5               # 早停耐心值

# GRPO Off-Policy 特有参数
grpo_num_gen=8             # 每个样本生成候选数量（add_gt=True时会自动减1给GT腾位置）
num_iterations=5           # 每批数据重复训练的轮数
steps_per_generation=1     # 每次生成的样本分成多少批
epsilon_low=0.2            # Clipping 下界
epsilon_high=0.2           # Clipping 上界

# ========== 路径配置 ==========
BASE_OUTPUT_DIR="/home/notebook/data/group/RLGR/output_beauty_ut"
DATA_DIR="./data/Beauty"
# 预训练模型路径（你的最优模型）
PRETRAINED_MODEL="${BASE_OUTPUT_DIR}/generation_model/lr3e-3_wd0.3"

# ========== 调参范围 ==========
# 后训练学习率（与 NDCG 脚本对齐）
LR_LIST=(1e-4)
# KL散度系数（与 NDCG 脚本对齐）
BETA_LIST=(1)
# PrefixMatchReward 特有：幂律前缀奖励参数
GAMMA_LIST=(0.3 0.4 0.5)       # 最大部分奖励上界
ALPHA_LIST=(0.8)           # 形状参数，α<1 倾向前期 token

# ========== 最优结果跟踪 ==========
best_ndcg10=0
best_exp_name=""
best_hit1=0
best_hit5=0
best_hit10=0
best_ndcg1=0
best_ndcg5=0

echo "=============================================="
echo "Beauty数据集 GRPO Off-Policy 后训练 (Prefix奖励)"
echo "=============================================="
echo "GPU数量: ${NUM_GPUS}"
echo "总批次大小: ${model_bsz} (每卡: $((model_bsz / NUM_GPUS)))"
echo "训练轮数: ${num_epochs}"
echo "评估频率: 每${eval_epoch}个epoch（从第${eval_start_epoch}个epoch后开始）"
echo "早停耐心值: ${early_stop}"
echo "学习率范围: ${LR_LIST[*]}"
echo "KL散度系数范围: ${BETA_LIST[*]}"
echo "Prefix gamma范围: ${GAMMA_LIST[*]}"
echo "Prefix alpha范围: ${ALPHA_LIST[*]}"
echo "奖励函数: Prefix (PrefixMatchReward with NDCG)"
echo "GRPO num_generations: ${grpo_num_gen} (add_gt=True时自动生成$((grpo_num_gen-1))个 + 1个GT = 总共${grpo_num_gen}个)"
echo "Off-Policy 参数:"
echo "  - num_iterations: ${num_iterations}"
echo "  - steps_per_generation: ${steps_per_generation}"
echo "  - epsilon_low: ${epsilon_low}"
echo "  - epsilon_high: ${epsilon_high}"
echo "  - add_gt: True (添加 Ground Truth)"
echo "  - generate_every: $((steps_per_generation * num_iterations))"
echo "预训练模型: ${PRETRAINED_MODEL}"
echo "=============================================="
echo ""
echo "固定参数："
echo "  - weight_decay: ${model_decay}"
echo "  - warmup_ratio: 0.05 (默认)"
echo "  - use_user_tokens: True (使用user tokens)"
echo "=============================================="
echo ""

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

# 总实验数
total_exp=$((${#LR_LIST[@]} * ${#BETA_LIST[@]} * ${#GAMMA_LIST[@]} * ${#ALPHA_LIST[@]}))
current_exp=0

# 用于存储所有实验结果的数组
declare -a all_results

# 遍历 lr、beta、gamma、alpha
for model_lr in "${LR_LIST[@]}"; do
    for grpo_beta in "${BETA_LIST[@]}"; do
        for prefix_gamma in "${GAMMA_LIST[@]}"; do
            for prefix_alpha in "${ALPHA_LIST[@]}"; do
                current_exp=$((current_exp + 1))
                
                # 构建实验名称和保存路径（lr、beta、gamma、alpha）
                exp_name="lr${model_lr}_beta${grpo_beta}_gamma${prefix_gamma}_alpha${prefix_alpha}"
                posttrain_save_path="${PRETRAINED_MODEL}/offline_prefix/${exp_name}"
                
                echo ""
                echo "=============================================="
                echo "实验 ${current_exp}/${total_exp}: ${exp_name}"
                echo "=============================================="
                echo "学习率: ${model_lr}"
                echo "KL散度系数: ${grpo_beta}"
                echo "Prefix gamma: ${prefix_gamma}"
                echo "Prefix alpha: ${prefix_alpha}"
                echo "奖励函数: Prefix (PrefixMatchReward with NDCG)"
                echo "模型保存路径: ${posttrain_save_path}"
                echo "=============================================="
                
                # Prefix 奖励配置（带 NDCG 惩罚，幂律参数 gamma、alpha）
                reward_config="online_rl.trainer.reward_func._target_=genrec.rewards.prefix_match_reward.PrefixMatchReward +online_rl.trainer.reward_func.use_ndcg=True +online_rl.trainer.reward_func.ndcg_weight=0.5 +online_rl.trainer.reward_func.num_tokens_per_item=4 +online_rl.trainer.reward_func.gamma=${prefix_gamma} +online_rl.trainer.reward_func.alpha=${prefix_alpha}"
    
                # 检查是否已经训练过（如果存在 HuggingFace 模型文件则跳过）
                if [ -f "${posttrain_save_path}/config.json" ]; then
                    echo "发现已存在的模型，跳过训练..."
                else
                    # 运行后训练（Off-Policy版本，使用user_tokens）
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
                        +model.eval_start_epoch=${eval_start_epoch} \
                        model.early_stop_upper_steps=${early_stop} \
                        model.use_user_tokens=True \
                        online_rl.trainer._target_=genrec.trainers.online_rl.grpo_off_policy_trainer.OffPolicyTrainer \
                        online_rl.trainer.beta=${grpo_beta} \
                        online_rl.trainer.num_generations=${grpo_num_gen} \
                        +online_rl.trainer.num_iterations=${num_iterations} \
                        +online_rl.trainer.steps_per_generation=${steps_per_generation} \
                        +online_rl.trainer.epsilon_low=${epsilon_low} \
                        +online_rl.trainer.epsilon_high=${epsilon_high} \
                        +online_rl.trainer.add_gt=True \
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
                
                hit1=$(grep "Hit@1:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*Hit@1: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
                hit5=$(grep "Hit@5:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*Hit@5: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
                hit10=$(grep "Hit@10:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*Hit@10: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
                ndcg1=$(grep "NDCG@1:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*NDCG@1: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
                ndcg5=$(grep "NDCG@5:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*NDCG@5: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
                ndcg10=$(grep "NDCG@10:" "$log_file" | grep -v "eval_" | tail -1 | sed 's/.*NDCG@10: \([0-9.]*\).*/\1/' | tr -d '\n\r ')
                
                if [ -z "$ndcg10" ]; then
                    echo "⚠️ 无法解析测试结果，跳过"
                    continue
                fi
                
                echo "📊 测试集结果:"
                echo "   Hit@1:  ${hit1}, Hit@5:  ${hit5}, Hit@10:  ${hit10}"
                echo "   NDCG@1: ${ndcg1}, NDCG@5: ${ndcg5}, NDCG@10: ${ndcg10}"
                
                all_results+=("${exp_name}|${hit1}|${hit5}|${hit10}|${ndcg1}|${ndcg5}|${ndcg10}")
                
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
                
                echo ""
                echo "实验 ${exp_name} 完成！"
                echo ""
            done
        done
    done
done

echo ""
echo "=============================================="
echo "📊 所有后训练实验结果汇总"
echo "=============================================="
printf "%-35s %-10s %-10s %-10s %-10s %-10s %-10s\n" "实验" "Hit@1" "Hit@5" "Hit@10" "NDCG@1" "NDCG@5" "NDCG@10"
echo "--------------------------------------------------------------------------------------------------------------"
for result in "${all_results[@]}"; do
    IFS='|' read -r name h1 h5 h10 n1 n5 n10 <<< "$result"
    printf "%-35s %-10s %-10s %-10s %-10s %-10s %-10s\n" "$name" "$h1" "$h5" "$h10" "$n1" "$n5" "$n10"
done

echo ""
echo "=============================================="
echo "🏆 最优后训练实验（基于测试集 NDCG@10）"
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
    echo "模型路径: ${PRETRAINED_MODEL}/offline_prefix/${best_exp_name}"
else
    echo "⚠️ 没有成功完成的实验"
fi
echo "=============================================="

echo ""
echo "后训练模型保存位置："
for model_lr in "${LR_LIST[@]}"; do
    for grpo_beta in "${BETA_LIST[@]}"; do
        for prefix_gamma in "${GAMMA_LIST[@]}"; do
            for prefix_alpha in "${ALPHA_LIST[@]}"; do
                exp_name="lr${model_lr}_beta${grpo_beta}_gamma${prefix_gamma}_alpha${prefix_alpha}"
                echo "  - ${PRETRAINED_MODEL}/offline_prefix/${exp_name}/"
            done
        done
    done
done
echo "=============================================="
