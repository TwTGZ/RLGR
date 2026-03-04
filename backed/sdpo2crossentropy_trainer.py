from collections import defaultdict    
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union    
import torch    
import torch.nn as nn    
import torch.nn.functional as F    

from torch.utils.data import Dataset  
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments    
from transformers.trainer_callback import TrainerCallback    
from transformers.modeling_outputs import BaseModelOutput  

# 🔴 导入 Trie 相关
from genrec.generation.trie import Trie, prefix_allowed_tokens_fn  

class SDPOTrainer(Trainer):    
    """S-DPO Trainer for Encoder-Decoder models with generation evaluation support"""    
        
    def __init__(  
        self,  
        model: Union[PreTrainedModel, nn.Module] = None,  
        ref_model: Union[PreTrainedModel, nn.Module] = None,  
        beta: float = 0.1,  
        args: TrainingArguments = None,  
        data_collator: Optional[DataCollator] = None,  
        eval_data_collator: Optional[DataCollator] = None,  
        label_pad_token_id: int = -100,  
        padding_value: int = 0,  
        train_dataset: Optional[Dataset] = None,  
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,  
        tokenizer: Optional[PreTrainedTokenizerBase] = None,  
        model_init: Optional[Callable[[], PreTrainedModel]] = None,  
        callbacks: Optional[List[TrainerCallback]] = None,  
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),  
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,  
        # 🔴 用于 evaluation 的参数  
        compute_metrics: Optional[Callable] = None,  
        generation_params: Optional[Dict] = None,  
        item2tokens: Optional[Dict] = None,  
        pad_token_id: Optional[int] = None,  
        eos_token_id: Optional[int] = None,  
        **kwargs,  
    ):  
        self.label_pad_token_id = label_pad_token_id  
        self.padding_value = padding_value  
        self.beta = beta  
        self.ref_model = ref_model  
        self._stored_metrics = defaultdict(lambda: defaultdict(list))  
        self.eval_data_collator = eval_data_collator  
          
        # 🔴 保存 evaluation 相关参数  
        self.generation_params = generation_params or {}  
        self.item2tokens = item2tokens  
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.eos_token_id = eos_token_id if eos_token_id is not None else 1
        
        # 🔴 构建 Trie（用于约束生成）
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        else:
            self.prefix_allowed_fn = None
            print("⚠️ 警告: 未提供 item2tokens，无法使用前缀约束生成。")
          
        super().__init__(  
            model=model,  
            args=args,  
            data_collator=data_collator,  
            train_dataset=train_dataset,  
            eval_dataset=eval_dataset,  
            tokenizer=tokenizer,  
            model_init=model_init,  
            compute_metrics=compute_metrics,  # 🔴 传递给父类
            callbacks=callbacks,  
            optimizers=optimizers,  
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,  
        )  
          
        if hasattr(self, "accelerator"):  
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)  
        else:  
            raise AttributeError("Trainer does not have an accelerator object")  
      
    def concatenated_forward(    
        self,    
        model: nn.Module,    
        batch: Dict[str, torch.Tensor]    
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:    
        """    
        优化版 Encoder-Decoder 前向传播：Encoder 只计算一次  
            
        返回:    
            chosen_logps: [B]    
            rejected_logps: [B, N]    
            chosen_logits: [B, L, V]    
            rejected_logits: [B, N, L, V]    
        """    
        batch_size = batch["input_ids"].shape[0]    
        num_rejected = batch["rejected_labels"].shape[1]    
            
        # ===== 1. Encoder 只计算一次 =====  
        encoder_outputs = model.encoder(    
            input_ids=batch["input_ids"],    
            attention_mask=batch["attention_mask"],    
            return_dict=True,    
        )    
          
        # ===== 2. Decoder for Chosen =====    
        chosen_outputs = model(    
            encoder_outputs=encoder_outputs,    
            attention_mask=batch["attention_mask"],    
            labels=batch["chosen_labels"],    
        )    
        chosen_logits = chosen_outputs.logits.to(torch.float32)  # [B, L, V]    
            
        # 计算 chosen 的 log probabilities    
        chosen_logps = self._get_batch_logps(    
            chosen_logits,    
            batch["chosen_labels"],    
            average_log_prob=False,    
        )  # [B,L]    
            

        return chosen_logps, chosen_logits    

 
    def get_batch_metrics(    
        self,    
        model,    
        batch: Dict[str, torch.Tensor],    
        train_eval: Literal["train", "eval"] = "train",    
    ):    
        """计算 batch 的 loss 和 metrics"""    
        metrics = {}    
            
        # Policy model 前向传播    
        (    
            policy_chosen_logps,      # [B]    
                # [B, N]    
            policy_chosen_logits     # [B, L, V]    
               # [B, N, L, V]    
        ) = self.concatenated_forward(model, batch)    
        
        losses = -policy_chosen_logps
 
            
        metrics[f"losses"] = losses.detach().cpu().numpy().mean()
 
        return losses.mean(), metrics

    def _get_batch_logps(    
        self,    
        logits: torch.FloatTensor,    
        labels: torch.LongTensor,    
        average_log_prob: bool = False,    
    ) -> torch.FloatTensor:    
        """计算给定 labels 在 logits 下的 log probabilities"""    
        if logits.shape[:-1] != labels.shape:    
            raise ValueError("Logits and labels must have the same shape (except last dim)")    
            
        # 创建labels的副本，避免in-place修改
        labels_clone = labels.clone()
        
        # Mask: 忽略 label_pad_token_id    
        loss_mask = labels_clone != self.label_pad_token_id    
            
        # 将 pad token 替换为 0（在副本上操作）    
        labels_clone[labels_clone == self.label_pad_token_id] = 0    
            
        # 计算每个 token 的 log probability    
        per_token_logps = torch.gather(    
            logits.log_softmax(-1),    
            dim=2,    
            index=labels_clone.unsqueeze(2)    
        ).squeeze(2)    
            
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)    
        else:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)     

    def compute_loss(  
        self,  
        model: Union[PreTrainedModel, nn.Module],  
        inputs: Dict[str, Union[torch.Tensor, Any]],  
        return_outputs=False,  
        num_items_in_batch=None,  
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:  
        """训练时调用 - 使用 S-DPO loss"""  
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")  
          
        if self.accelerator.is_main_process:  
            self.store_metrics(metrics, train_eval="train")  
          
        if return_outputs:  
            return (loss, metrics)  
        return loss  
      
    def prediction_step(  
        self,  
        model: Union[PreTrainedModel, nn.Module],  
        inputs: Dict[str, Union[torch.Tensor, Any]],  
        prediction_loss_only: bool,  
        ignore_keys: Optional[List[str]] = None,  
    ):  
        """
        评估时调用 - 始终使用生成式评估
        忽略 rejected_labels，只对 chosen_labels 进行生成预测
        """  
        if ignore_keys is None:  
            if hasattr(model, "config"):  
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])  
            else:  
                ignore_keys = []  
      
        # ===== 准备输入 =====
        inputs = self._prepare_inputs(inputs)
        
        # 🔴 获取 chosen_labels（优先）或 labels
        has_labels = "chosen_labels" in inputs or "labels" in inputs
        labels = inputs.get("chosen_labels", inputs.get("labels"))
        
        # ===== 1. 计算损失（基于 chosen_labels）=====
        with torch.no_grad():
            if has_labels:
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
        
        # ===== 2. 执行生成操作 =====
        gen_kwargs = {
            "max_length": self.generation_params.get('max_gen_length', 5),
            "num_beams": self.generation_params.get('num_beams', 10),
            "num_return_sequences": self.generation_params.get('max_k', 10),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        # 🔴 添加前缀约束（如果有）
        if self.prefix_allowed_fn:
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_fn
        
        unwrapped_model = self.accelerator.unwrap_model(model)
        generated_sequences = unwrapped_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        
        # ===== 3. Reshape 生成结果 =====
        # (batch_size * num_beams, seq_len) -> (batch_size, num_beams, seq_len)
        batch_size = inputs["input_ids"].shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        
        # ===== 4. 返回结果 =====
        # (loss, predictions, labels)
        # predictions: 生成的序列 [B, num_beams, L]
        # labels: chosen_labels（用于 compute_metrics）
        return (loss, generated_ids_reshaped, labels)

          
    def store_metrics(    
        self,    
        metrics: Dict[str, float],    
        train_eval: Literal["train", "eval"] = "train"    
    ) -> None:    
        """存储 metrics"""    
        for key, value in metrics.items():    
            self._stored_metrics[train_eval][key].append(value)    

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:  
        """Log metrics to the various objects watching training"""  
        train_eval = "train" if "loss" in logs else "eval"  
          
        # Add stored metrics to logs  
        if train_eval in self._stored_metrics:  
            for key, metrics in self._stored_metrics[train_eval].items():  
                if len(metrics) > 0:  
                    logs[key] = torch.tensor(metrics).mean().item()  
              
            # Clear stored metrics for this phase  
            self._stored_metrics[train_eval].clear()  
          
        # Call parent's log method  
        return super().log(logs, *args, **kwargs)  
      
    def get_eval_dataloader(self, eval_dataset=None):  
        """重写以使用不同的 collator"""  
        if eval_dataset is None:  
            eval_dataset = self.eval_dataset  
          
        # 临时替换 collator  
        original_collator = self.data_collator  
        if self.eval_data_collator is not None:  
            self.data_collator = self.eval_data_collator  
          
        dataloader = super().get_eval_dataloader(eval_dataset)  
          
        # 恢复原 collator  
        self.data_collator = original_collator  
          
        return dataloader