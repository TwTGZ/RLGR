# genrec/trainers/online_rl/grpo_off_policy_trainer.py

from typing import Callable, Optional, Dict, List, Tuple, Sized  # ✅ Sized 在 typing 中
import torch
from torch.utils.data import Sampler, DataLoader
from transformers import T5ForConditionalGeneration, TrainerCallback

from .base_trainer import BaseOnlineRLTrainer

class RepeatSampler(Sampler):
    """
    Custom sampler for off-policy training.
    Repeats samples in chunks to enable multiple gradient steps per generation.
    """
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        # Use same randomization as default DataLoader
        if self.shuffle:
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        # If repeat_count=1 and mini_repeat_count=1, return directly (same as default behavior)
        if self.repeat_count == 1 and self.mini_repeat_count == 1:
            for index in indexes:
                yield index
            return

        # For repeat > 1, keep original logic
        chunks = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        
        if self.drop_last:
            chunks = [chunk for chunk in chunks if len(chunk) == self.batch_size]

        # Repeat by chunk
        for chunk in chunks:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        if self.drop_last:
            num_complete_batches = self.num_samples // self.batch_size
            return num_complete_batches * self.batch_size * self.mini_repeat_count * self.repeat_count
        else:
            return self.num_samples * self.mini_repeat_count * self.repeat_count
class OffPolicyTrainer(BaseOnlineRLTrainer):
    """
    Off-Policy GRPO Trainer with importance sampling and clipping.
    
    Modified to use fixed total sample size, split across steps_per_generation.
    """
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        ref_model: T5ForConditionalGeneration,
        beta: float,
        num_generations: int,
        args=None,
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
        # Off-policy specific parameters
        num_iterations: int = 1,
        steps_per_generation: int = 1,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
    ):
        # Initialize base class
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            num_generations=num_generations,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            generation_params=generation_params,
            reward_func=reward_func,
            item2tokens=item2tokens,
            tokens2item=tokens2item,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            optimizers=optimizers,
        )
        
        # ===== Off-policy specific parameters =====
        self.num_iterations = num_iterations
        self.steps_per_generation = steps_per_generation
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.generate_every = steps_per_generation * num_iterations
        
        # ===== Off-policy state =====
        self._step = 0
        self._buffered_inputs = None
    
    def _get_train_sampler(self):
        """
        Return off-policy RepeatSampler.
        
        Key change: batch_size is NOT multiplied by steps_per_generation.
        We generate fixed number of samples, then split them.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=1,
            # ✅ 修改：不再乘以 steps_per_generation
            batch_size=self.args.per_device_train_batch_size,
            repeat_count=self.generate_every,
            shuffle=True,
            seed=self.args.seed,
            drop_last=False
        )
    
    def get_train_dataloader(self):
        """
        Override to use custom RepeatSampler.
        
        Key change: batch_size is NOT multiplied by steps_per_generation.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        dataloader_params = {
            # ✅ 修改：不再乘以 steps_per_generation
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": self._get_train_sampler(),
        }
        
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def _prepare_inputs_for_training(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Off-policy specific input preparation.
        
        Key logic:
        1. Check if need to regenerate (every generate_every steps)
        2. If yes, generate new completions and split into mini-batches
        3. Return the corresponding mini-batch from buffer
        """
        
        # Check if need to regenerate
        if self._step % self.generate_every == 0 or self._buffered_inputs is None:
            # Generate new completions
            generation_batch = self._generate_and_score_completions(inputs)
            
            # ✅ 修改：分割成 steps_per_generation 个小批次
            # 现在 generation_batch 的大小是固定的 (batch_size * num_generations)
            generation_batches = self._split_tensor_dict(generation_batch, self.steps_per_generation)
            self._buffered_inputs = generation_batches
        
        # Get corresponding batch from buffer
        batch_idx = self._step % self.steps_per_generation
        return self._buffered_inputs[batch_idx]
    
    def _split_tensor_dict(self, tensor_dict: Dict[str, torch.Tensor], num_splits: int) -> List[Dict[str, torch.Tensor]]:
        """
        Split tensor dict into multiple small batches.
        
        ✅ 修改：现在正确地分割固定大小的 batch
        """
        # 获取第一个张量的 batch 维度
        batch_size = next(iter(tensor_dict.values())).size(0)
        
        # 确保可以整除
        if batch_size % num_splits != 0:
            raise ValueError(
                f"Batch size {batch_size} cannot be evenly divided by "
                f"steps_per_generation {num_splits}. "
                f"Please ensure per_device_train_batch_size * num_generations "
                f"is divisible by steps_per_generation."
            )
        
        split_size = batch_size // num_splits
        splits = []
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size
            split = {k: v[start_idx:end_idx] for k, v in tensor_dict.items()}
            splits.append(split)
        
        return splits
    
    def _generate_and_score_completions(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate completions and compute rewards, advantages, old_logprobs.
        
        ✅ 不需要修改：输入的 batch_size 已经是固定的
        """
        device = self.accelerator.device
        
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        target_labels = inputs["labels"].to(device)
        
        total_batch_size = encoder_input_ids.size(0)
        num_beams = self.num_generations
        
        # ===== Calculate generation counts =====
        if self.add_gt:
            num_gt_per_sample = 1
            num_generated = num_beams - num_gt_per_sample
        else:
            num_gt_per_sample = 0
            num_generated = num_beams
        
        # ===== Generate sequences =====
        if num_generated > 0:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    max_length=self.max_completion_length,

                    do_sample = True,
                    # num_beams=num_generated, # 删除
                    num_return_sequences=num_generated,
                    temperature = 1.0,


                    early_stopping=True,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    decoder_start_token_id=self.decoder_start_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    prefix_allowed_tokens_fn=self.prefix_allowed_fn,
                )
            generated_ids = outputs.sequences[:, 1:]
            generated_ids = torch.cat([
                generated_ids,
                torch.ones_like(generated_ids[:, :1])
            ], dim=1)
        else:
            generated_ids = None
        
        # ===== Add ground truth =====
        if self.add_gt and num_gt_per_sample > 0:
            gt_decoder_ids = target_labels.repeat_interleave(num_gt_per_sample, dim=0)
        
        # ===== Merge generated and GT =====
        if self.add_gt and num_gt_per_sample > 0:
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
                
                generated_ids_reshaped = generated_ids.view(total_batch_size, num_generated, max_len)
                gt_decoder_ids_reshaped = gt_decoder_ids.view(total_batch_size, num_gt_per_sample, max_len)
                all_decoder_ids = torch.cat([generated_ids_reshaped, gt_decoder_ids_reshaped], dim=1)
                all_decoder_ids = all_decoder_ids.view(total_batch_size * num_beams, max_len)
            else:
                all_decoder_ids = gt_decoder_ids
        else:
            all_decoder_ids = generated_ids
        
        generated_ids = all_decoder_ids
        
        # ===== Mask after EOS =====
        is_eos = generated_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # ===== Compute rewards =====
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
        
        rewards = self.reward_func(
            generated_items,
            target_items,
            num_generations=self.num_generations
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # ===== Compute advantages using base class method =====
        def compute_offpolicy_advantages(gathered: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Compute off-policy advantages (same as GRPO)."""
            gathered_rewards = gathered["rewards"]
            total_samples = gathered_rewards.size(0) // num_beams
            
            rewards_reshaped = gathered_rewards.view(total_samples, num_beams)
            mean_grouped = rewards_reshaped.mean(dim=1, keepdim=True)
            std_grouped = rewards_reshaped.std(dim=1, keepdim=True)
            advantages = (rewards_reshaped - mean_grouped) / (std_grouped + 1e-5)
            return advantages.view(-1)
        
        advantages, gathered_data = self._gather_compute_slice(
            tensors_to_gather={"rewards": rewards},
            batch_size=total_batch_size,
            num_seqs_per_sample=num_beams,
            compute_fn=compute_offpolicy_advantages,
            return_gathered=True,
        )
        
        gathered_rewards = gathered_data["rewards"]
        
        # ===== Expand encoder inputs =====
        encoder_input_ids_expanded = encoder_input_ids.repeat_interleave(num_beams, dim=0)
        encoder_attention_mask_expanded = encoder_attention_mask.repeat_interleave(num_beams, dim=0)
        
        # ===== Compute old log probs (for importance sampling) =====
        with torch.no_grad():
            old_outputs = self.model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                labels=generated_ids,
                return_dict=True,
            )
            old_logits = old_outputs.logits
            
            labels_clone = generated_ids.clone()
            loss_mask = labels_clone != self.pad_token_id
            labels_clone[labels_clone == self.pad_token_id] = 0
            
            old_per_token_logps = torch.gather(
                old_logits.log_softmax(-1),
                dim=2,
                index=labels_clone.unsqueeze(-1)
            ).squeeze(-1)
            
            # ✅ 添加：立即释放 logits 内存
            del old_logits
        
        # ===== Compute reference log probs =====
        with torch.no_grad():
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
                index=labels_clone.unsqueeze(-1)
            ).squeeze(-1)
            
            # ✅ 添加：立即释放 logits 内存
            del ref_logits
        
        # ===== Log metrics =====
        self._metrics["reward"].append(gathered_rewards.mean().item())
        
        total_samples = gathered_rewards.size(0) // num_beams
        rewards_reshaped = gathered_rewards.view(total_samples, num_beams)
        self._metrics["reward_std"].append(rewards_reshaped.std(dim=1).mean().item())
        
        if self.add_gt and num_gt_per_sample > 0:
            gt_rewards = rewards_reshaped[:, -num_gt_per_sample:].mean()
            self._metrics["gt_reward"].append(gt_rewards.item())
            
            if num_generated > 0:
                gen_rewards = rewards_reshaped[:, :num_generated].mean()
                self._metrics["gen_reward"].append(gen_rewards.item())
        
        unique_items = len(set([item for item in generated_items if item != -1]))
        total_items = len([item for item in generated_items if item != -1])
        diversity = unique_items / total_items if total_items > 0 else 0.0
        self._metrics["diversity"].append(diversity)
        
        correct = sum([1 for gen, tgt in zip(generated_items, target_items) if gen == tgt and gen != -1])
        accuracy = correct / len(generated_items) if len(generated_items) > 0 else 0.0
        self._metrics["accuracy"].append(accuracy)
        
        return {
            "encoder_input_ids": encoder_input_ids_expanded,
            "encoder_attention_mask": encoder_attention_mask_expanded,
            "decoder_input_ids": generated_ids,
            "completion_mask": completion_mask,
            "loss_mask": loss_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute off-policy GRPO loss with importance sampling and clipping.
        
        ✅ 不需要修改
        """
        if return_outputs:
            raise ValueError("OffPolicyTrainer does not support returning outputs")
        
        encoder_input_ids = inputs["encoder_input_ids"]
        encoder_attention_mask = inputs["encoder_attention_mask"]
        decoder_input_ids = inputs["decoder_input_ids"]
        loss_mask = inputs["loss_mask"]
        old_per_token_logps = inputs["old_per_token_logps"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]

        # Forward pass
        outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            labels=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits
        
        # Compute new log probs
        labels_clone = decoder_input_ids.clone()
        labels_clone[labels_clone == self.pad_token_id] = 0
        
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels_clone.unsqueeze(-1)
        ).squeeze(-1)
        
        # ===== Off-policy: Importance Sampling =====
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(log_ratio)
        
        # ===== Off-policy: Clipping =====
        ratio_clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        
        # ===== GRPO Loss with clipping =====
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = ratio_clipped * advantages.unsqueeze(1)
        per_token_policy_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # ===== KL Divergence =====
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # ===== Total Loss =====
        per_token_loss = per_token_policy_loss + self.beta * per_token_kl
        
        # Average over tokens and batch
        loss = (per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)
        
        # ===== Log metrics =====
        mean_kl = ((per_token_kl * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)).mean()
        mean_policy_loss = ((per_token_policy_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)).mean()
        
        # Clipping ratio
        is_clipped = ((ratio < 1 - self.epsilon_low) | (ratio > 1 + self.epsilon_high)) & loss_mask.bool()
        clip_ratio = is_clipped.float().sum() / loss_mask.sum().clamp(min=1.0)
        
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl.detach()).mean().item())
        self._metrics["policy_loss"].append(self.accelerator.gather_for_metrics(mean_policy_loss.detach()).mean().item())
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio.detach()).mean().item())
        
        # Log importance sampling ratio
        mean_ratio = (ratio * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        self._metrics["importance_ratio"].append(self.accelerator.gather_for_metrics(mean_ratio.detach()).mean().item())
        
        return loss.mean()
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to update step counter.
        
        ✅ 不需要修改
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return loss