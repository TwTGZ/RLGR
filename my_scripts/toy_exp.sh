#!/bin/bash
# Sports数据集二阶段调参测试脚本 - 玩具版本
# 用于测试代码是否能正常运行，以及文件结构是否正确
# 2x2=4组实验，10个epoch

# 切换到项目根目录
cd /home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark

# ========== GPU配置 ==========
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# ========== 测试参数（快速验证用） ==========
model_bsz=1024
num_epochs=2              # 只跑10个epoch
eval_epoch=2               # 每5个epoch评估一次（即只评估2次）
early_stop=5

# ========== 路径配置 ==========
BASE_OUTPUT_DIR="/home/notebook/code/personal/S9059888/Generative-Recommendation-Benchmark/output_Official/output_sports"
DATA_DIR="./data/Sports"

# ========== 调参范围（2x2=4组） ==========
LR_LIST=(1e-3)
WD_LIST=(0.1)

# ========== 最优结果跟踪 ==========
best_ndcg10=0
best_exp_name=""
best_hit1=0
best_hit5=0
best_hit10=0
best_ndcg1=0
best_ndcg5=0

echo "=============================================="
echo "🧪 测试脚本：验证调参框架"
echo "=============================================="
echo "GPU数量: ${NUM_GPUS}"
echo "总批次大小: ${model_bsz}"
echo "训练轮数: ${num_epochs}"
echo "学习率范围: ${LR_LIST[*]}"
echo "权重衰减范围: ${WD_LIST[*]}"
echo "输出目录: ${BASE_OUTPUT_DIR}"
echo "=============================================="

# 检查tokenizer是否存在
if [ ! -f "${BASE_OUTPUT_DIR}/tokenizer_model/tokenizer.pkl" ]; then
    echo "错误: Tokenizer不存在于 ${BASE_OUTPUT_DIR}/tokenizer_model/"
    exit 1
fi

total_exp=$((${#LR_LIST[@]} * ${#WD_LIST[@]}))
current_exp=0

# 用于存储所有实验结果的数组
declare -a all_results

for model_lr in "${LR_LIST[@]}"; do
    for model_decay in "${WD_LIST[@]}"; do
        current_exp=$((current_exp + 1))
        
        exp_name="lr${model_lr}_wd${model_decay}"
        model_save_path="${BASE_OUTPUT_DIR}/generation_model/${exp_name}"
        
        echo ""
        echo "=============================================="
        echo "🧪 测试实验 ${current_exp}/${total_exp}: ${exp_name}"
        echo "=============================================="
        
        # 检查是否已训练过
        if [ -f "${model_save_path}/Sports_final_model.pt" ]; then
            echo "发现已存在的模型，跳过训练..."
        else
            accelerate launch --num_processes=${NUM_GPUS} train_with_generative.py \
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
                generative.save_model_path="${model_save_path}"
        fi
        
        # ========== 解析测试集结果 ==========
        # 查找该实验的日志文件（取最新的）
        log_file=$(find "${model_save_path}/logs" -name "*.log" -type f 2>/dev/null | head -1)
        
        if [ -z "$log_file" ]; then
            echo "⚠️ 未找到日志文件，跳过结果解析"
            continue
        fi
        
        echo "📋 解析日志: ${log_file}"
        
        # 解析测试集结果（格式：Hit@1: 0.0034, NDCG@1: 0.0034）
        # 使用 grep -v "eval_" 排除验证集结果（验证集格式是 eval_hit@1:）
        # 使用 tr -d '\n\r ' 清理隐藏字符
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
        
        # 更新最优结果（基于NDCG@10，与早停指标一致）
        # 使用 awk 进行浮点数比较（比 bc 更健壮）
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
printf "%-20s %-10s %-10s %-10s %-10s %-10s %-10s\n" "实验" "Hit@1" "Hit@5" "Hit@10" "NDCG@1" "NDCG@5" "NDCG@10"
echo "--------------------------------------------------------------------------------------------------------------"
for result in "${all_results[@]}"; do
    IFS='|' read -r name h1 h5 h10 n1 n5 n10 <<< "$result"
    printf "%-20s %-10s %-10s %-10s %-10s %-10s %-10s\n" "$name" "$h1" "$h5" "$h10" "$n1" "$n5" "$n10"
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
    echo "模型路径: ${BASE_OUTPUT_DIR}/generation_model/${best_exp_name}"
else
    echo "⚠️ 没有成功完成的实验"
fi
echo "=============================================="

# 展示实际生成的目录
echo ""
echo "实际生成的目录："
ls -la "${BASE_OUTPUT_DIR}/generation_model/" 2>/dev/null || echo "目录不存在"
