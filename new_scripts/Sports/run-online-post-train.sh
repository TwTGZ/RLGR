#!/bin/bash
# Sports数据集的在线RL后训练（Post-Training）
# 使用GRPO算法进行在线强化学习微调
# 单GPU版本

cd ../..
export CUDA_VISIBLE_DEVICES=0

# ========== 训练参数 ==========
model_lr=2e-3              # 学习率
model_decay=0.3            # 权重衰减
model_bsz=512              # 批次大小
online_beta=1              # GRPO的KL散度系数
online_negnum=4            # 每个样本生成的候选数量
num_epochs=25               # 训练轮数

# ========== 路径配置 ==========
# 注意：output_dir 要指向包含 tokenizer_model 的目录
BASE_DIR="./output/output_sports_multigpu"
PRETRAINED_MODEL="$BASE_DIR/generation_model"
POSTTRAIN_SAVE="./output/output_sports_posttrain/posttrain_model"

echo "=================================="
echo "开始Sports数据集在线RL后训练"
echo "使用单GPU训练"
echo "=================================="
echo "学习率: $model_lr"
echo "权重衰减: $model_decay"
echo "批次大小: $model_bsz"
echo "KL系数(beta): $online_beta"
echo "候选生成数: $online_negnum"
echo "训练轮数: $num_epochs"
echo "预训练模型: $PRETRAINED_MODEL"
echo "后训练保存: $POSTTRAIN_SAVE"
echo "=================================="

python train_with_online_rl.py \
    skip_tokenizer=True \
    tokenizer.data_text_files="./data/Sports/item2title.pkl" \
    tokenizer.interaction_files="./data/Sports/user2item.pkl" \
    output_dir="$BASE_DIR" \
    model.data_interaction_files="./data/Sports/user2item.pkl" \
    model.data_text_files="./data/Sports/item2title.pkl" \
    model.learning_rate=$model_lr \
    model.weight_decay=$model_decay \
    model.batch_size=$model_bsz \
    model.num_epochs=$num_epochs \
    online_rl.trainer.beta=$online_beta \
    online_rl.trainer.num_generations=$online_negnum \
    online_rl.pretrained_model="$PRETRAINED_MODEL" \
    online_rl.save_model_path="$POSTTRAIN_SAVE"

echo "=================================="
echo "后训练完成！"
echo "模型保存位置: $POSTTRAIN_SAVE"
echo "=================================="