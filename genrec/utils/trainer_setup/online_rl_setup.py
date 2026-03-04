# genrec/utils/trainer_setup/online_rl/online_rl_setup.py

import os
from typing import Optional, Dict, List, Callable
from functools import partial
from transformers import TrainingArguments, EarlyStoppingCallback
from hydra.utils import instantiate
from omegaconf import DictConfig

from genrec.utils.metrics import compute_metrics
from genrec.utils.callbacks.generative.generative_callback import (
    GenerativeLoggingCallback,
    DelayedEvaluateEveryNEpochsCallback
)
from genrec.utils.models_setup.conditional_t5_setup import create_t5_model

def setup_training(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    model_config,
    online_rl_config: DictConfig,
    output_dirs,
    logger,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    train_data_collator,
    custom_output_dir=None,  # 【新增】自定义输出目录（用于checkpoint和日志）
):
    """
    统一的 Online RL 训练设置函数
    
    Args:
        custom_output_dir: 自定义输出目录（用于checkpoint和日志），为None则使用默认
    """
    
    # ============【新增】确定输出目录 ============
    if custom_output_dir:
        checkpoint_dir = custom_output_dir
        log_dir = os.path.join(custom_output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
    else:
        checkpoint_dir = output_dirs['model']
        log_dir = output_dirs['logs']
    
    # ===== 1. 训练参数配置 =====
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,  # 【修改】使用自定义目录
        num_train_epochs=model_config['num_epochs'],
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,  # 禁用自动加载，避免多卡同步问题，改为手动加载
        logging_dir=log_dir,  # 【修改】使用自定义目录
        logging_steps=100,
        report_to=[],
        warmup_ratio=model_config["warmup_ratio"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        metric_for_best_model="ndcg@10",
        greater_is_better=True,
    )
    
    # ===== 2. 生成评估参数 =====
    tokens_to_item_map = tokenizer.tokens2item
    compute_metrics_with_map = partial(
        compute_metrics,
        tokens_to_item_map=tokens_to_item_map
    )
    
    num_beams = model_config.get('num_beams', 10)
    max_gen_length = model_config.get('max_gen_length', 5)
    k_list = model_config.get('k_list', [5, 10, 20])
    max_k = k_list[-1] if k_list else 10
    
    generation_params = {
        'max_gen_length': max_gen_length,
        'num_beams': num_beams,
        'max_k': max_k
    }
    
    # ===== 3. 回调函数 =====
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=model_config.get("early_stop_upper_steps", 1000)
        ),
        GenerativeLoggingCallback(logger),
        DelayedEvaluateEveryNEpochsCallback(
            n_epochs=model_config.get("evaluation_epoch", 5),
            start_epoch=model_config.get("eval_start_epoch", 0)
        )
    ]
    
    # ===== 4. 创建参考模型 =====
    # logger.info("创建参考模型（Reference Model）...")
    # ref_model = create_t5_model(
    #     vocab_size=tokenizer.vocab_size,
    #     model_config=model_config
    # )
    # ref_model.load_state_dict(model.state_dict())
    ref_model = create_t5_model(
        vocab_size=model.config.vocab_size, 
        model_config=model_config
    )
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    # logger.info("参考模型创建完成")
    
    # ===== 5. 创建奖励函数（如果配置中有）=====
    reward_func = None
    if 'reward_func' in online_rl_config.trainer:
        # logger.info(f"实例化 Reward Function: {online_rl_config.trainer.reward_func._target_}")
        reward_func = instantiate(online_rl_config.trainer.reward_func)
        # logger.info("Reward Function 创建完成")
    
    # ===== 6. 使用 partial instantiate 创建 Trainer =====
    # logger.info(f"实例化 Trainer: {online_rl_config.trainer._target_}")
    
    # 🔥 使用 instantiate 获取 partial 函数
    trainer_partial = instantiate(online_rl_config.trainer)
    
    # 🔥 调用 partial 函数，传入运行时参数
    trainer = trainer_partial(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics_with_map,
        generation_params=generation_params,
        item2tokens=tokenizer.item2tokens,
        tokens2item=tokenizer.tokens2item,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        reward_func=reward_func,
    )
    
    # logger.info(f"Trainer 配置完成:")
    # logger.info(f"  - Trainer 类型: {online_rl_config.trainer._target_}")
    # logger.info(f"  - Beta: {online_rl_config.trainer.get('beta', 'N/A')}")
    # logger.info(f"  - Num generations: {online_rl_config.trainer.get('num_generations', 'N/A')}")
    # if reward_func:
        # logger.info(f"  - Reward Function: {type(reward_func).__name__}")
    # logger.info(f"  - Num beams: {num_beams}")
    # logger.info(f"  - Max gen length: {max_gen_length}")
    # logger.info(f"  - Max k: {max_k}")
    
    return trainer