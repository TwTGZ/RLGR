# genrec/trainers/online_rl/rankpo_off_policy_trainer.py

from typing import Callable, Optional, Dict, List, Tuple, Sized
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

        # If repeat_count=1 and mini_repeat_count=1, return directly
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

class RankPOOffPolicyTrainer(BaseOnlineRLTrainer):
    """
    Off-Policy RankPO Trainer with importance sampling and clipping.
    
    Key differences from online RankPO:
    - Uses RepeatSampler to repeat samples for multiple gradient steps
    - Buffers generated completions and reuses them
    - Applies importance sampling with ratio clipping
    """
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        ref_model: T5ForConditionalGeneration,
        beta: float,
        tau1: float,  # RankPO-specific parameter
        tau2: float,  # RankPO-specific parameter
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
        
        # ===== RankPO-specific parameters =====
        self.tau1 = tau1
        self.tau2 = tau2
        self.activation = lambda x, tau: torch.where(
            x < 0, 
            torch.exp(x/tau) / (2*tau), 
            torch.exp(-x/tau) / (2*tau)
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
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=1,
            batch_size=self.args.per_device_train_batch_size,
            repeat_count=self.generate_every,
            shuffle=True,
            seed=self.args.seed,
            drop_last=False
        )
    
    def get_train_dataloader(self):
        """
        Override to use custom RepeatSampler.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        dataloader_params = {
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
            
            # Split into steps_per_generation mini-batches
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
        Generate completions and compute scores, advantages, old_logprobs.
        
        This is RankPO-specific, using quantile-based advantages.
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
        
        # ===== Generate sequences with scores =====
        if num_generated > 0:
            with torch.no_grad():
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
                generated_ids = outputs.sequences[:, 1:]
                generated_ids = torch.cat([
                    generated_ids,
                    torch.full_like(generated_ids[:, :1], self.eos_token_id)
                ], dim=1)
                generated_scores = outputs.sequences_scores
        else:
            generated_ids = None
            generated_scores = None
        
        # ===== Reshape =====
        if generated_ids is not None:
            seq_len = generated_ids.size(1)
            generated_ids = generated_ids.view(total_batch_size, num_generated, seq_len)
            generated_scores = generated_scores.view(total_batch_size, num_generated)
        
        # ===== Compute quantile =====
        if generated_scores is not None:
            K = num_generated
            quantiles = generated_scores[:, K-1]  # [B]
        else:
            quantiles = None
        
        # ===== Add ground truth with its score =====
        if self.add_gt and num_gt_per_sample > 0:
            with torch.no_grad():
                gt_outputs = self.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    labels=target_labels,
                    return_dict=True,
                )
                gt_logits = gt_outputs.logits
                
                gt_log_probs = torch.gather(
                    gt_logits.log_softmax(-1),
                    dim=2,
                    index=target_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                gt_mask = (target_labels != self.pad_token_id).float()
                gt_sequence_scores = (gt_log_probs * gt_mask).sum(dim=1)
            
            gt_ids = target_labels.unsqueeze(1)  # [B, 1, L]
            gt_sequence_scores = gt_sequence_scores.unsqueeze(1)  # [B, 1]
            
            # Merge
            if generated_ids is not None:
                max_len = max(generated_ids.size(2), gt_ids.size(2))
                
                if generated_ids.size(2) < max_len:
                    padding = torch.full(
                        (generated_ids.size(0), generated_ids.size(1), max_len - generated_ids.size(2)),
                        self.pad_token_id,
                        dtype=generated_ids.dtype,
                        device=device
                    )
                    generated_ids = torch.cat([generated_ids, padding], dim=2)
                
                if gt_ids.size(2) < max_len:
                    padding = torch.full(
                        (gt_ids.size(0), gt_ids.size(1), max_len - gt_ids.size(2)),
                        self.pad_token_id,
                        dtype=gt_ids.dtype,
                        device=device
                    )
                    gt_ids = torch.cat([gt_ids, padding], dim=2)
                
                all_ids = torch.cat([generated_ids, gt_ids], dim=1)  # [B, num_generated+1, L]
                all_scores = torch.cat([generated_scores, gt_sequence_scores], dim=1)  # [B, num_generated+1]
            else:
                all_ids = gt_ids
                all_scores = gt_sequence_scores
            
            num_seqs_per_sample = num_generated + 1 if generated_ids is not None else 1
        else:
            all_ids = generated_ids
            all_scores = generated_scores
            num_seqs_per_sample = num_generated
        
        # ===== Flatten =====
        seq_len = all_ids.size(2)
        all_ids_flat = all_ids.view(-1, seq_len)  # [B * num_seqs, L]
        all_scores_flat = all_scores.view(-1)  # [B * num_seqs]
        
        # ===== Mask after EOS =====
        is_eos = all_ids_flat == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # ===== Get all item IDs =====
        all_items = []
        for i in range(all_ids_flat.size(0)):
            tokens = all_ids_flat[i].cpu().tolist()
            item = self._tokens_to_item(tokens)
            all_items.append(item if item is not None else -1)
        all_items = torch.tensor(all_items, device=device)
        
        # GT item IDs
        gt_items_list = []
        for i in range(total_batch_size):
            tokens = target_labels[i].cpu().tolist()
            item = self._tokens_to_item(tokens)
            gt_items_list.append(item if item is not None else -1)
        gt_items_expanded = torch.tensor(
            [gt_items_list[i // num_seqs_per_sample] for i in range(total_batch_size * num_seqs_per_sample)],
            device=device
        )
        
        # Positive/negative labels
        is_positive = (all_items == gt_items_expanded).float()
        
        # ===== Compute advantages using base class method =====
        def compute_rankpo_advantages(gathered: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Compute RankPO advantages based on quantiles and positive labels."""
            gathered_scores = gathered["scores"]
            gathered_is_positive = gathered["is_positive"]
            gathered_quantiles = gathered["quantiles"]
            
            total_samples = gathered_scores.size(0) // num_seqs_per_sample

            scores_reshaped = gathered_scores.view(total_samples, num_seqs_per_sample)
            is_positive_reshaped = gathered_is_positive.view(total_samples, num_seqs_per_sample)
            quantiles_reshaped = gathered_quantiles.view(total_samples)
            
            # Compute delta and advantages
            exceeding = (scores_reshaped - quantiles_reshaped.unsqueeze(1))
            
            all_score = exceeding
            pos_score = -exceeding[:, -1].unsqueeze(-1)

            all_delta = self.activation(all_score, self.tau1)
            pos_delta = torch.sigmoid(pos_score)

            pos_advantage = is_positive_reshaped.float()
            delta_sum = all_delta.sum(dim=1, keepdim=True)
            neg_advantage = -(all_delta / (delta_sum + 1e-8)) * (1 - is_positive_reshaped)

            advantages = ((pos_advantage + neg_advantage) * pos_delta)
            advantages = advantages.view(-1)
            return advantages
        
        advantages, gathered_data = self._gather_compute_slice(
            tensors_to_gather={
                "scores": all_scores_flat,
                "is_positive": is_positive,
                "quantiles": quantiles,
            },
            batch_size=total_batch_size,
            num_seqs_per_sample=num_seqs_per_sample,
            compute_fn=compute_rankpo_advantages,
            return_gathered=True,
        )
        
        gathered_scores = gathered_data["scores"]
        gathered_is_positive = gathered_data["is_positive"]
        gathered_quantiles = gathered_data["quantiles"]
        
        # ===== Expand encoder inputs =====
        encoder_input_ids_expanded = encoder_input_ids.repeat_interleave(num_seqs_per_sample, dim=0)
        encoder_attention_mask_expanded = encoder_attention_mask.repeat_interleave(num_seqs_per_sample, dim=0)
        
        # ===== Compute old log probs (for importance sampling) =====
        with torch.no_grad():
            old_outputs = self.model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                labels=all_ids_flat,
                return_dict=True,
            )
            old_logits = old_outputs.logits
            
            labels_clone = all_ids_flat.clone()
            loss_mask = labels_clone != self.pad_token_id
            labels_clone[labels_clone == self.pad_token_id] = 0
            
            old_per_token_logps = torch.gather(
                old_logits.log_softmax(-1),
                dim=2,
                index=labels_clone.unsqueeze(-1)
            ).squeeze(-1)
            
            # Release memory
            del old_logits
        
        # ===== Compute reference log probs =====
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                labels=all_ids_flat,
                return_dict=True,
            )
            ref_logits = ref_outputs.logits
            
            ref_per_token_logps = torch.gather(
                ref_logits.log_softmax(-1),
                dim=2,
                index=labels_clone.unsqueeze(-1)
            ).squeeze(-1)
            
            # Release memory
            del ref_logits
        
        # ===== Log metrics =====
        self._metrics["mean_score"].append(gathered_scores.mean().item())
        self._metrics["quantile"].append(gathered_quantiles.mean().item())
        self._metrics["advantage_mean"].append(advantages.mean().item())
        self._metrics["advantage_std"].append(advantages.std().item())
        
        if gathered_is_positive.sum() > 0:
            pos_scores = gathered_scores[gathered_is_positive.bool()]
            self._metrics["pos_score_mean"].append(pos_scores.mean().item())
        
        if (1 - gathered_is_positive).sum() > 0:
            neg_scores = gathered_scores[(1 - gathered_is_positive).bool()]
            self._metrics["neg_score_mean"].append(neg_scores.mean().item())
        
        accuracy = gathered_is_positive.mean().item()
        self._metrics["accuracy"].append(accuracy)
        
        unique_items = len(set(all_items.cpu().tolist()))
        total_items = (all_items != -1).sum().item()
        diversity = unique_items / total_items if total_items > 0 else 0.0
        self._metrics["diversity"].append(diversity)
        
        return {
            "encoder_input_ids": encoder_input_ids_expanded,
            "encoder_attention_mask": encoder_attention_mask_expanded,
            "decoder_input_ids": all_ids_flat,
            "completion_mask": completion_mask,
            "loss_mask": loss_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute off-policy RankPO loss with importance sampling and clipping.
        """
        if return_outputs:
            raise ValueError("RankPOOffPolicyTrainer does not support returning outputs")
        
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
        
        # ===== RankPO Loss with clipping =====
        # Original RankPO: policy_scores * advantages
        # Off-policy: ratio * advantages (with clipping)
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
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return loss