cd ../..
export CUDA_VISIBLE_DEVICES=1
tokenizer_lr=1e-3
model_bsz=1024
python train_rqvae.py \
    output_dir="/home/zsj/pretrained-models/Sports-Official" \
    tokenizer.data_text_files="./data/Sports/item2title.pkl" \
    tokenizer.interaction_files="./data/Sports/user2item.pkl" \
    tokenizer.learning_rate=$tokenizer_lr \
    tokenizer.batch_size=$model_bsz \
    tokenizer.epochs=20000 
    
    
    
