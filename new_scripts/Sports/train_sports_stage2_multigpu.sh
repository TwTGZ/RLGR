#!/bin/bash
# 训练Sports数据集的二阶段T5生成模型 - 2卡版本
# 使用已有的tokenizer，跳过一阶段训练

# 切换到项目根目录（从 new_scripts/Sports/ 返回两级）
cd ../..

# ========== GPU配置 ==========
export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和GPU 1
NUM_GPUS=2

# ========== 训练参数 ==========
model_lr=2e-3              # 学习率
model_decay=0.3            # 权重衰减
model_bsz=1024             # 总批次大小（会自动分配到每张卡，每卡512）
num_epochs=100             # 训练轮数
eval_epoch=5               # 每5个epoch评估一次

echo "=================================="
echo "开始训练Sports数据集二阶段模型"
echo "使用 ${NUM_GPUS} 张GPU分布式训练"
echo "=================================="
echo "学习率: $model_lr"
echo "权重衰减: $model_decay"
echo "总批次大小: $model_bsz (每卡: $((model_bsz / NUM_GPUS)))"
echo "训练轮数: $num_epochs"
echo "评估频率: 每${eval_epoch}个epoch"
echo "使用的tokenizer: ./output/output_sports_multigpu/tokenizer/"
echo "=================================="

accelerate launch --num_processes=${NUM_GPUS} train_with_generative.py \
    skip_tokenizer=True \
    tokenizer.data_text_files="./data/Sports/item2title.pkl" \
    tokenizer.interaction_files="./data/Sports/user2item.pkl" \
    output_dir="./output/output_sports_multigpu" \
    model.data_interaction_files="./data/Sports/user2item.pkl" \
    model.data_text_files="./data/Sports/item2title.pkl" \
    model.learning_rate=$model_lr \
    model.weight_decay=$model_decay \
    model.batch_size=$model_bsz \
    model.num_epochs=$num_epochs \
    model.evaluation_epoch=$eval_epoch

echo "=================================="
echo "训练完成！"
echo "模型保存位置: ./output/output_sports_multigpu/generation_model/"
echo "=================================="