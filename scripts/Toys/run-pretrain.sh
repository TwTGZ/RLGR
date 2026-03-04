cd ../..
export CUDA_VISIBLE_DEVICES=2 
model_lr=3e-3
model_decay=0.6
model_bsz=1024
python train_with_generative.py \
    tokenizer.data_text_files="./data/Toys/item2title.pkl" \
    tokenizer.interaction_files="./data/Toys/user2item.pkl" \
    output_dir="/home/zsj/pretrained-models/Toys-Official" \
    model.data_interaction_files="./data/Toys/user2item.pkl" \
    model.data_text_files="./data/Toys/item2title.pkl" \
    model.learning_rate=$model_lr \
    model.weight_decay=$model_decay \
    model.batch_size=$model_bsz
