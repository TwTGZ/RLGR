# genrec/utils/trainer_setup/generative/generative_setup.py

import os
from typing import Optional, Dict, List
from functools import partial
from transformers import TrainingArguments, EarlyStoppingCallback
from hydra.utils import instantiate
from omegaconf import DictConfig

from genrec.utils.metrics import compute_metrics
from genrec.utils.callbacks.generative.generative_callback import (
    GenerativeLoggingCallback,
    EvaluateEveryNEpochsCallback,
    DelayedEvaluateEveryNEpochsCallback
)

def setup_training(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    model_config,
    generative_config: DictConfig,  # 新增：generative 配置
    output_dirs,
    logger,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    train_data_collator,
    custom_output_dir=None,  # 自定义输出目录（用于checkpoint和日志）
):
    """
    统一的 Generative 训练设置函数
    
    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
        model_config: 模型配置
        generative_config: Generative 配置（包含 trainer 配置）
        output_dirs: 输出目录
        logger: 日志记录器
        per_device_train_batch_size: 训练批次大小
        per_device_eval_batch_size: 评估批次大小
        train_data_collator: 训练数据 collator
        custom_output_dir: 自定义输出目录（用于checkpoint和日志），为None则使用默认
    """
    
    # ============ 确定输出目录 ============
    if custom_output_dir:
        checkpoint_dir = custom_output_dir
        log_dir = os.path.join(custom_output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
    else:
        checkpoint_dir = output_dirs['model']
        log_dir = output_dirs['logs']
    
    # ===== 1. 训练参数配置 =====
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=model_config['num_epochs'],
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=log_dir,
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
        EvaluateEveryNEpochsCallback(
            n_epochs=model_config.get("evaluation_epoch", 5)
        ),
        # DelayedEvaluateEveryNEpochsCallback(n_epochs=model_config.get("evaluation_epoch", 5), start_epoch=120)
    ]
    
    # ===== 4. 使用 partial instantiate 创建 Trainer =====
    # logger.info(f"实例化 Trainer: {generative_config.trainer._target_}")
    
    # 🔥 使用 instantiate 获取 partial 函数
    trainer_partial = instantiate(generative_config.trainer)
    
    # 🔥 调用 partial 函数，传入运行时参数
    trainer = trainer_partial(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics_with_map,
        generation_params=generation_params,
        item2tokens=tokenizer.item2tokens,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
    )
    
    # logger.info(f"Trainer 配置完成:")
    # logger.info(f"  - Trainer 类型: {generative_config.trainer._target_}")
    # logger.info(f"  - Num beams: {num_beams}")
    # logger.info(f"  - Max gen length: {max_gen_length}")
    # logger.info(f"  - Max k: {max_k}")
    # logger.info(f"  - Metric for best model: {training_args.metric_for_best_model}")
    
    return trainer