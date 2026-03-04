cd ../..
export CUDA_VISIBLE_DEVICES=0
model_lr=2e-3
model_decay=0.3
model_bsz=1024
python train_with_generative.py \
    tokenizer.data_text_files="./data/Sports/item2title.pkl" \
    tokenizer.interaction_files="./data/Sports/user2item.pkl" \
    output_dir="/home/zsj/pretrained-models/Sports-Official" \
    model.data_interaction_files="./data/Sports/user2item.pkl" \
    model.data_text_files="./data/Sports/item2title.pkl" \
    model.learning_rate=$model_lr \
    model.weight_decay=$model_decay \
    model.batch_size=$model_bsz
