cd ../..
export CUDA_VISIBLE_DEVICES=4 
tokenizer_lr=1e-3
model_bsz=1024
model_rl=1e-3
model_beta=1e-2
model_negnum=4 
python train_with_offline_rl.py \
    output_dir="/home/zsj/pretrained-models/s-Test" \
    tokenizer.data_text_files="./data/Beauty/item2title.pkl" \
    tokenizer.interaction_files="./data/Beauty/user2item.pkl" \
    tokenizer.learning_rate=$tokenizer_lr \
    tokenizer.batch_size=$model_bsz \
    model.data_interaction_files="./data/Beauty/user2item.pkl" \
    model.data_text_files="./data/Beauty/item2title.pkl" \
    model.num_epochs=10 \
    offline_rl.beta=$model_beta \
    offline_rl.neg_num=$model_negnum \
    
    
    
