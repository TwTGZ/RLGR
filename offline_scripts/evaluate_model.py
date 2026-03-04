#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单独的模型测试脚本
用于测试已训练但未完成测试评估的模型

使用方法:
    python offline_scripts/evaluate_model.py \
        --model_path /path/to/model \
        --tokenizer_path /path/to/tokenizer.pkl \
        --data_dir ./data/Toys \
        --use_user_tokens

注意: 建议使用单 GPU 运行以避免多卡同步问题
    CUDA_VISIBLE_DEVICES=0 python offline_scripts/evaluate_model.py ...
"""

import os
import sys
import argparse
import torch
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genrec.quantization.tokenizers.rqvae_tokenizer import RQVAETokenizer
from genrec.data.datasets.generative.tiger_dataset import TigerDataset
from genrec.data.collators.generative.tiger_collator import TigerDataCollator
from genrec.utils.evaluation_utils import evaluate_model_with_constrained_beam_search
from genrec.utils.common_utils import set_seed


def setup_logging(log_dir: str = None):
    """设置日志"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（可选）
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'evaluate_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")
    
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='评估已训练的模型')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（HuggingFace 格式目录）')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Tokenizer 路径（tokenizer.pkl 文件）')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录（包含 item2title.pkl 和 user2item.pkl）')
    
    # 可选参数
    parser.add_argument('--use_user_tokens', action='store_true',
                        help='是否使用 user tokens')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='测试批次大小 (default: 256)')
    parser.add_argument('--max_seq_len', type=int, default=20,
                        help='最大序列长度 (default: 20)')
    parser.add_argument('--num_beams', type=int, default=10,
                        help='Beam search 数量 (default: 10)')
    parser.add_argument('--max_gen_length', type=int, default=5,
                        help='最大生成长度 (default: 5)')
    parser.add_argument('--k_list', type=int, nargs='+', default=[1, 5, 10],
                        help='评估的 K 值列表 (default: 1 5 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志保存目录（默认不保存到文件）')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # ========== 检查路径 ==========
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.tokenizer_path):
        logger.error(f"Tokenizer 路径不存在: {args.tokenizer_path}")
        return
    
    data_interaction_files = os.path.join(args.data_dir, 'user2item.pkl')
    data_text_files = os.path.join(args.data_dir, 'item2title.pkl')
    
    if not os.path.exists(data_interaction_files):
        logger.error(f"交互数据文件不存在: {data_interaction_files}")
        return
    
    if not os.path.exists(data_text_files):
        logger.error(f"文本数据文件不存在: {data_text_files}")
        return
    
    # ========== 加载 Tokenizer ==========
    logger.info(f"加载 Tokenizer: {args.tokenizer_path}")
    tokenizer = RQVAETokenizer.load(args.tokenizer_path)
    logger.info(f"Tokenizer 词汇表大小: {tokenizer.vocab_size}")
    logger.info(f"物品数量: {len(tokenizer.item2tokens)}")
    
    # ========== 加载模型 ==========
    logger.info(f"加载模型: {args.model_path}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 创建数据集配置 ==========
    model_config = {
        'max_seq_len': args.max_seq_len,
        'use_user_tokens': args.use_user_tokens,
        'test_batch_size': args.batch_size,
    }
    
    logger.info(f"数据集配置:")
    logger.info(f"  - max_seq_len: {args.max_seq_len}")
    logger.info(f"  - use_user_tokens: {args.use_user_tokens}")
    logger.info(f"  - batch_size: {args.batch_size}")
    
    # ========== 创建测试数据集 ==========
    logger.info("创建测试数据集...")
    test_dataset = TigerDataset(
        data_interaction_files=data_interaction_files,
        data_text_files=data_text_files,
        tokenizer=tokenizer,
        config=model_config,
        mode='test'
    )
    logger.info(f"测试集样本数: {len(test_dataset)}")
    
    # ========== 创建 DataLoader ==========
    test_data_collator = TigerDataCollator(
        max_seq_len=test_dataset.max_token_len,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        mode='test'
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_data_collator
    )
    
    # ========== 使用简单的单 GPU 评估 ==========
    logger.info("="*50)
    logger.info("开始测试评估（单 GPU 模式）...")
    logger.info(f"  - num_beams: {args.num_beams}")
    logger.info(f"  - max_gen_length: {args.max_gen_length}")
    logger.info(f"  - k_list: {args.k_list}")
    logger.info("="*50)
    
    # 创建一个简化的 accelerator-like 对象用于单 GPU
    class SimpleAccelerator:
        def __init__(self, device):
            self.device = device
        
        @property
        def is_main_process(self):
            return True
        
        def gather_for_metrics(self, tensor):
            return tensor
        
        def wait_for_everyone(self):
            pass
        
        def unwrap_model(self, model):
            # 单 GPU 模式下直接返回模型本身
            return model
    
    simple_accelerator = SimpleAccelerator(device)
    
    # 执行评估（predictions 保存到模型目录，避免多次评估覆盖）
    output_json_path = os.path.join(args.model_path, "predictions.json")
    evaluate_model_with_constrained_beam_search(
        model=model,
        eval_dataloader=test_dataloader,
        accelerator=simple_accelerator,
        tokenizer=tokenizer,
        k_list=args.k_list,
        num_beams=args.num_beams,
        max_gen_length=args.max_gen_length,
        logger=logger,
        mode="Test",
        output_json_path=output_json_path,
    )
    
    logger.info("="*50)
    logger.info("✅ 测试评估完成!")
    logger.info("="*50)


if __name__ == '__main__':
    main()
