
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if return_outputs:
        raise ValueError("The GRPOTrainerForGenRec does not support returning outputs")
    
    encoder_input_ids = inputs["encoder_input_ids"]
    encoder_attention_mask = inputs["encoder_attention_mask"]
    decoder_input_ids = inputs["decoder_input_ids"]
    completion_mask = inputs["completion_mask"]

    # Forward pass
    outputs = model(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        labels=decoder_input_ids,
        return_dict=True,
    )
    logits = outputs.logits  # [B*num_beams, L, vocab_size]
    # ✅ 直接对齐（T5 已经处理了 shifting）
    
    shifted_labels = decoder_input_ids   # 去掉第一个 token（通常是 <pad> 或 <bos>）
    shifted_logits = logits   # 去掉最后一个 token 的预测


    labels_clone    = shifted_labels.clone()
    # Mask: 忽略 label_pad_token_id    
    loss_mask = labels_clone != 1
    # 将 pad token 替换为 0（在副本上操作）    
    labels_clone[labels_clone == 1] = 0    




    per_token_logps = torch.gather(
        logits.log_softmax(-1),    
        dim=2,
        index=labels_clone.unsqueeze(-1)
    ).squeeze(-1)  # [B*num_beams, L]
    

    loss = (per_token_logps * loss_mask).sum(-1)   / loss_mask.sum(-1)    


    return -loss.mean()