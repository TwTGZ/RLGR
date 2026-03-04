# genrec/rewards/prefix_match_reward.py

import math
from typing import List, Optional
from .base_reward import BaseReward


class PrefixMatchReward(BaseReward):
    """
    Power-law prefix reward with optional NDCG penalty.
    
    Formula: r_prefix(k) = γ * (k/(T-1))^α  for partial match (k<T)
             r_prefix(T) = 1.0              for full match (k=T)
             r_prefix(0) = 0.0             when first token is wrong
    
    - γ (gamma): Upper bound of max partial reward (k=T-1 reaches γ)
    - α (alpha): Shape. α<1 concave (earlier tokens weighted more), α=1 linear, α>1 convex
    - T = num_tokens_per_item
    
    Example (T=4, γ=0.5, α=0.5):
        k=0 (first wrong): 0.00
        k=1: 0.5*(1/3)^0.5 ≈ 0.29
        k=2: 0.5*(2/3)^0.5 ≈ 0.41
        k=3: 0.5*1^0.5 = 0.50 (max partial)
        k=4 (full): 1.00
    """
    
    def __init__(
        self, 
        use_ndcg: bool = False, 
        ndcg_weight: float = 0.5,
        num_tokens_per_item: int = 4,
        gamma: float = 0.5,
        alpha: float = 0.5,
    ):
        """
        Initialize PrefixMatchReward.
        
        Args:
            use_ndcg: Whether to add NDCG-based ranking penalty
            ndcg_weight: Weight for NDCG penalty (0-1)
            num_tokens_per_item: T, number of tokens per item
            gamma: Max partial reward upper bound (k=T-1 reaches γ). Controls magnitude.
            alpha: Shape parameter. α<1: concave, earlier tokens get larger increments.
                   α=1: linear. α>1: convex, later tokens get larger increments.
        """
        self.use_ndcg = use_ndcg
        self.ndcg_weight = ndcg_weight
        self.num_tokens_per_item = num_tokens_per_item
        self.gamma = gamma
        self.alpha = alpha
    
    def _compute_prefix_match_score(
        self, 
        gen_tokens: List[int], 
        target_tokens: List[int]
    ) -> float:
        """
        Compute power-law prefix match score.
        
        r(k) = γ * (k/(T-1))^α for k in [1, T-1]
        r(T) = 1.0 for full match
        r(0) = 0.0 when first token wrong
        """
        if not gen_tokens or not target_tokens:
            return 0.0
        
        min_len = min(len(gen_tokens), len(target_tokens))
        T = self.num_tokens_per_item
        
        matched = 0
        for i in range(min_len):
            if gen_tokens[i] == target_tokens[i]:
                matched += 1
            else:
                break
        
        # 第一个 token 就错时显式设为 0
        if matched == 0:
            return 0.0
        
        # 完整匹配 k=T
        if matched >= T:
            return 1.0
        
        # 部分匹配 k in [1, T-1]: r = γ * (k/(T-1))^α
        if T <= 1:
            return 1.0
        
        k = matched
        denom = T - 1
        return self.gamma * ((k / denom) ** self.alpha)

    def __call__(
        self, 
        generated_items: List[int], 
        target_items: List[int],
        num_generations: int = None,
        generated_tokens: Optional[List[List[int]]] = None,
        target_tokens: Optional[List[List[int]]] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute prefix match rewards.
        
        Args:
            generated_items: List of generated item IDs (used for exact match check)
            target_items: List of target item IDs
            num_generations: Number of generations per sample (required if use_ndcg=True)
            generated_tokens: List of generated token sequences (required for prefix matching)
            target_tokens: List of target token sequences (required for prefix matching)
            **kwargs: Unused
        
        Returns:
            List of rewards based on prefix matching
        """
        # 检查是否有 token 信息
        if generated_tokens is None or target_tokens is None:
            # 回退到简单的二元匹配（兼容性）
            return [1.0 if g == t else 0.0 for g, t in zip(generated_items, target_items)]
        
        if not self.use_ndcg:
            # 不使用 NDCG，直接计算前缀匹配分数
            return [
                self._compute_prefix_match_score(gen, tgt) 
                for gen, tgt in zip(generated_tokens, target_tokens)
            ]
        
        # ===== 使用 NDCG 惩罚（与 GRPOReward 类似）=====
        if num_generations is None:
            raise ValueError("num_generations is required when use_ndcg=True")
        
        # 预计算 NDCG penalties
        ndcg_penalties = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
        ndcg_sum = sum(ndcg_penalties)
        ndcg_penalties = [-elm / ndcg_sum for elm in ndcg_penalties]
        
        rewards = []
        
        # 按组处理
        for group_idx in range(len(generated_items) // num_generations):
            start_idx = group_idx * num_generations
            
            for rank in range(num_generations):
                idx = start_idx + rank
                
                # 计算前缀匹配分数
                prefix_score = self._compute_prefix_match_score(
                    generated_tokens[idx], target_tokens[idx]
                )
                
                # 是否完全匹配（正样本）
                is_positive = (generated_items[idx] == target_items[idx])
                
                if is_positive:
                    # 正样本：前缀分数为 1.0，NDCG 惩罚为 0
                    final_reward = (1 - self.ndcg_weight) * 1.0 + self.ndcg_weight * 0.0
                else:
                    # 负样本：前缀分数 + NDCG 惩罚
                    final_reward = (1 - self.ndcg_weight) * prefix_score + self.ndcg_weight * ndcg_penalties[rank]
                
                rewards.append(final_reward)
        
        return rewards
