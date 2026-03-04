cd ../..
export CUDA_VISIBLE_DEVICES=2 
tokenizer_lr=1e-3 
model_bsz=1024 
python train_rqvae.py \
    output_dir="/home/zsj/pretrained-models/Toys-Official" \
    tokenizer.data_text_files="./data/Toys/item2title.pkl" \
    tokenizer.interaction_files="./data/Toys/user2item.pkl" \
    tokenizer.learning_rate=$tokenizer_lr \
    tokenizer.batch_size=$model_bsz \
    tokenizer.epochs=20000 
    
    
    
