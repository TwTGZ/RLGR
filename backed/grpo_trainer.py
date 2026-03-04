
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Dict, List, Tuple

import torch.nn as nn    

import torch
from torch.utils.data import Sampler
from accelerate.utils import gather
from transformers import (
    Trainer,
    TrainerCallback,
    T5ForConditionalGeneration,
)

from transformers import PreTrainedModel, Trainer    
from genrec.generation.trie import Trie,prefix_allowed_tokens_fn




class GRPOTrainer(Trainer):
    """
    GRPO Trainer for Generative Recommendation with Encoder-Decoder models.
    """
    
    _tag_names = ["trl", "grpo", "genrec"]
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        ref_model,
        beta,
        num_generations,
        args = None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        callbacks: Optional[List[TrainerCallback]] = None,
        compute_metrics: Optional[Callable] = None,  
        generation_params: Optional[Dict] = None,  
        reward_func: Optional[Callable] = None,
        item2tokens: Optional[Dict] = None,  
        tokens2item: Optional[Dict] = None,  
        pad_token_id: Optional[int] = None,  
        eos_token_id: Optional[int] = None, 
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        num_iterations=5,  # 新增：每次生成后更新的次数,
        logger = None,
    ):
        
        # Get item2tokens from tokenizer
        self.item2tokens = item2tokens
        self.tokens2item = tokens2item
        
        
        # Build Trie for constrained generation
        self.candidate_trie = Trie(self.item2tokens)
        self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        
        # Training arguments

        
        self.num_generations = num_generations
        
        self.beta = beta
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = model.config.decoder_start_token_id
        self.generation_params = generation_params or {}  
        self.max_completion_length = self.generation_params.get('max_gen_length',5)
        self.logger = logger

        # Reward function
        self.reward_func = reward_func if reward_func else self._default_reward_func
        

        self.ref_model = ref_model
        
        # Initialize metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions if hasattr(args, 'log_completions') else False
        self.add_gt = True
        
        self.num_iterations = num_iterations  # 添加这个参数
            # 用于追踪当前是第几次迭代
        self._current_iteration = 0
        self._buffered_generation_batch = None  # 缓存生成的完整批次

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_metrics=compute_metrics
        )
                

        if hasattr(self, "accelerator"):  
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)  
        else:  
            raise AttributeError("Trainer does not have an accelerator object")  
      

        # Validation
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )


    def train(self, *args, **kwargs):
        """重写 train 方法，实现生成 + 多轮训练"""
        
        # ========== Phase 1: 生成数据 ==========
        self.logger.info("=" * 50)
        self.logger.info("Phase 1: Generating completions for all batches")
        self.logger.info("=" * 50)
        
        self.is_generation_phase = True
        self.model.eval()  # 生成时使用 eval 模式
        
        self.generated_data_cache = []
        
        # 遍历一遍数据集，生成所有 completions
        for step, inputs in enumerate(self.get_train_dataloader()):
            inputs = self._prepare_inputs(inputs)
            generated_batch = self._generate_completions(inputs)
            self.generated_data_cache.append(generated_batch)
            
            if step % 10 == 0:
                self.logger.info(f"Generated {step + 1} batches")
        
        self.logger.info(f"Total generated batches: {len(self.generated_data_cache)}")
        
        # ========== Phase 2: 多轮训练 ==========
        self.logger.info("=" * 50)
        self.logger.info(f"Phase 2: Training for {self.num_iterations} epochs")
        self.logger.info("=" * 50)
        
        self.is_generation_phase = False
        self.model.train()
        
        # 训练 N 个 epoch
        for epoch in range(self.num_iterations):
            self._current_training_epoch = epoch
            self.logger.info(f"\n--- Training Epoch {epoch + 1}/{self.num_iterations} ---")
            
            # 调用父类的训练逻辑
            output = super().train(*args, **kwargs)
            
            # 每个 epoch 后可以选择性地重新生成数据
            if epoch < self.num_iterations - 1:
                # 可选：每隔几个 epoch 重新生成数据
                if (epoch + 1) % 1 == 0:  # 例如每 5 个 epoch 重新生成
                    self.logger.info("Regenerating data...")
                    self.is_generation_phase = True
                    self.model.eval()
                    self.generated_data_cache = []
                    for step, inputs in enumerate(self.get_train_dataloader()):
                        inputs = self._prepare_inputs(inputs)
                        generated_batch = self._generate_completions(inputs)
                        self.generated_data_cache.append(generated_batch)
                    self.is_generation_phase = False
                    self.model.train()
        
        return output


    def _default_reward_func(self, generated_items: List[int], target_items: List[int]) -> List[float]:
        """
        Default reward function: 1.0 if generated item matches target, 0.0 otherwise.
        
        Args:
            generated_items: List of generated item IDs
            target_items: List of target item IDs
            
        Returns:
            List of rewards
        """
        rewards = []
        for gen_item, target_item in zip(generated_items, target_items):
            rewards.append(1.0 if gen_item == target_item else 0.0)
        return rewards

    def _tokens_to_item(self, token_list: List[int]) -> Optional[int]:
        """Convert a list of tokens to item ID."""
        # Remove padding and special tokens
        clean_tokens = [t for t in token_list if t not in [self.pad_token_id, self.eos_token_id, self.decoder_start_token_id]]
        tokens_tuple = tuple(clean_tokens)
        return self.tokens2item.get(tokens_tuple, None)
    
    def _prepare_inputs(self, inputs):
        """准备输入数据"""
        if not self.model.training:
            # 评估模式：使用标准输入
            return super()._prepare_inputs(inputs)
        
        if self.is_generation_phase:
            # 生成阶段：返回原始输入（用于生成）
            return inputs
        else:
            # 训练阶段：从缓存中获取对应的生成数据
            batch_idx = self.state.global_step % len(self.generated_data_cache)
            return self.generated_data_cache[batch_idx]
   
    def _generate_completions(self, inputs):
        """生成 completions 并计算 rewards（你原来的逻辑）"""
        device = self.accelerator.device
        
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        target_labels = inputs["labels"].to(device)
        
        batch_size = encoder_input_ids.size(0)
        num_beams = self.num_generations
        
        # 生成部分
        if self.add_gt:
            num_gt_per_sample = 1
            num_generated = num_beams - num_gt_per_sample
        else:
            num_gt_per_sample = 0
            num_generated = num_beams
        
        with torch.no_grad():
            if num_generated > 0:
                outputs = self.model.generate(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    max_length=self.max_completion_length,
                    num_beams=num_generated,
                    num_return_sequences=num_generated,
                    early_stopping=True,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    decoder_start_token_id=self.decoder_start_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    prefix_allowed_tokens_fn=self.prefix_allowed_fn,
                )
                generated_ids = outputs.sequences
            else:
                generated_ids = None
            
            # 添加 GT
            if self.add_gt and num_gt_per_sample > 0:
                gt_decoder_ids = target_labels.repeat_interleave(num_gt_per_sample, dim=0)
                
                if generated_ids is not None:
                    max_len = max(generated_ids.size(1), gt_decoder_ids.size(1))
                    
                    if generated_ids.size(1) < max_len:
                        padding = torch.full(
                            (generated_ids.size(0), max_len - generated_ids.size(1)),
                            self.pad_token_id,
                            dtype=generated_ids.dtype,
                            device=device
                        )
                        generated_ids = torch.cat([generated_ids, padding], dim=1)
                    
                    if gt_decoder_ids.size(1) < max_len:
                        padding = torch.full(
                            (gt_decoder_ids.size(0), max_len - gt_decoder_ids.size(1)),
                            self.pad_token_id,
                            dtype=gt_decoder_ids.dtype,
                            device=device
                        )
                        gt_decoder_ids = torch.cat([gt_decoder_ids, padding], dim=1)
                    
                    generated_ids_reshaped = generated_ids.view(batch_size, num_generated, max_len)
                    gt_decoder_ids_reshaped = gt_decoder_ids.view(batch_size, num_gt_per_sample, max_len)
                    all_decoder_ids = torch.cat([generated_ids_reshaped, gt_decoder_ids_reshaped], dim=1)
                    all_decoder_ids = all_decoder_ids.view(batch_size * num_beams, max_len)
                else:
                    all_decoder_ids = gt_decoder_ids
            else:
                all_decoder_ids = generated_ids
            
            generated_ids = all_decoder_ids
            
            # Mask after EOS
            is_eos = generated_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            
            # 计算 rewards
            generated_items = []
            target_items = []
            
            for i in range(generated_ids.size(0)):
                gen_tokens = generated_ids[i].cpu().tolist()
                gen_item = self._tokens_to_item(gen_tokens)
                generated_items.append(gen_item if gen_item is not None else -1)
                
                sample_idx = i // num_beams
                target_tokens = target_labels[sample_idx].cpu().tolist()
                target_item = self._tokens_to_item(target_tokens)
                target_items.append(target_item if target_item is not None else -1)
            
            rewards = self.reward_func(generated_items, target_items, num_generations=self.num_generations)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            rewards = gather(rewards)
            
            # 计算 advantages
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            
            # 计算 reference log probs
            encoder_input_ids_expanded = encoder_input_ids.repeat_interleave(num_beams, dim=0)
            encoder_attention_mask_expanded = encoder_attention_mask.repeat_interleave(num_beams, dim=0)
            
            ref_outputs = self.ref_model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                labels=generated_ids,
                return_dict=True,
            )
            ref_logits = ref_outputs.logits
            
            ref_per_token_logps = torch.gather(
                ref_logits.log_softmax(-1),
                dim=2,
                index=generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # 🔴 计算初始策略的 log probs（用于 importance sampling）
            initial_outputs = self.model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                labels=generated_ids,
                return_dict=True,
            )
            initial_logits = initial_outputs.logits
            
            labels_clone = generated_ids.clone()
            loss_mask = labels_clone != self.pad_token_id
            labels_clone[labels_clone == self.pad_token_id] = 0
            
            initial_per_token_logps = torch.gather(
                initial_logits.log_softmax(-1),
                dim=2,
                index=labels_clone.unsqueeze(-1)
            ).squeeze(-1)
        
        # 返回生成的数据
        return {
            "encoder_input_ids": encoder_input_ids_expanded,
            "encoder_attention_mask": encoder_attention_mask_expanded,
            "decoder_input_ids": generated_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "initial_per_token_logps": initial_per_token_logps,  # 🔴 保存初始策略的 logps
            "advantages": advantages,
            "sliced_rewards": rewards,
        }


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainerForGenRec does not support returning outputs")
        
        if self.is_generation_phase:
            # 生成阶段不计算 loss
            return torch.tensor(0.0, device=self.accelerator.device)
        
        encoder_input_ids = inputs["encoder_input_ids"]
        encoder_attention_mask = inputs["encoder_attention_mask"]
        decoder_input_ids = inputs["decoder_input_ids"]
        completion_mask = inputs["completion_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        initial_per_token_logps = inputs["initial_per_token_logps"]  # 🔴 初始策略
        advantages = inputs["advantages"]
        
        # 计算当前策略的 log probs
        outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            labels=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits
        
        labels_clone = decoder_input_ids.clone()
        loss_mask = labels_clone != self.pad_token_id
        labels_clone[labels_clone == self.pad_token_id] = 0
        
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels_clone.unsqueeze(-1)
        ).squeeze(-1)
        
        # 🔴 Importance Sampling：使用初始策略的 logps
        log_ratio = per_token_logps - initial_per_token_logps
        importance_ratio = torch.exp(log_ratio)
        
        # 截断以提高稳定性
        importance_ratio = torch.clamp(importance_ratio, max=2.0)
        
        # 计算 KL 散度
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # 计算 loss
        policy_scores = importance_ratio * advantages.unsqueeze(1)
        cross_entropy_loss = -policy_scores
        kl_divergence_loss = self.beta * per_token_kl
        per_token_loss = cross_entropy_loss + kl_divergence_loss
        
        loss = (per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1)
        
        # 记录指标
        mean_kl = ((per_token_kl * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean()
        mean_importance_ratio = ((importance_ratio * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean()
        
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl.detach()).mean().item())
        self._metrics["importance_ratio"].append(self.accelerator.gather_for_metrics(mean_importance_ratio.detach()).mean().item())
        self._metrics["training_epoch"].append(self._current_training_epoch)
        
        return loss.mean()

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        评估时调用 - 使用生成式评估
        """
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        # ===== 准备输入 =====
        inputs = self._prepare_inputs(inputs)
        
        # 获取 labels
        has_labels = "labels" in inputs
        labels = inputs.get("labels")
        
        # ===== 1. 计算损失（使用 GRPO 的 _prepare_inputs 和 compute_loss）=====
        with torch.no_grad():
            if has_labels:
                # 使用 GRPO 的完整流程计算 loss
                loss_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "labels": labels,
                }
                outputs = model(**loss_inputs)
                loss = outputs.loss.mean().detach() if outputs.loss is not None else torch.tensor(0.0)
            else:
                loss = torch.tensor(0.0)
        
        # 如果只需要 loss，直接返回
        if prediction_loss_only:
            return (loss, None, None)
        
        # ===== 2. 执行生成操作（用于评估指标）=====
        device = self.accelerator.device
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        
        # 生成参数
        gen_kwargs = {
            "max_length": self.generation_params.get('max_gen_length', 5),
            "num_beams": self.generation_params.get('num_beams', 10),
            "num_return_sequences": self.generation_params.get('max_k', 10),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,

            "num_return_sequences": self.num_generations,
            "decoder_start_token_id": self.decoder_start_token_id,
        }
        
        # 🔴 添加前缀约束（使用 GRPO 的 Trie）
        if hasattr(self, 'prefix_allowed_fn') and self.prefix_allowed_fn:
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_fn
        
        # 执行生成
        unwrapped_model = self.accelerator.unwrap_model(model)
        generated_sequences = unwrapped_model.generate(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            **gen_kwargs,
        )
        
        # ===== 3. Reshape 生成结果 =====
        # (batch_size * num_beams, seq_len) -> (batch_size, num_beams, seq_len)
        batch_size = encoder_input_ids.shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        
        # ===== 4. 返回结果 =====
        # (loss, predictions, labels)
        # predictions: 生成的序列 [B, num_beams, L]
        # labels: 原始 labels（用于 compute_metrics）
        return (loss, generated_ids_reshaped, labels)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()