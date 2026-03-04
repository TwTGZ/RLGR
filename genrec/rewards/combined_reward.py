# genrec/rewards/combined_reward.py

import math
from typing import List, Optional
from .base_reward import BaseReward


class CombinedReward(BaseReward):
    """
    Combined reward: PrefixMatchReward (reward part) + CalibrationReward (penalty part).
    
    - Reward: Power-law prefix match score r(k) = γ * (k/(T-1))^α (partial) or 1.0 (full match)
    - Penalty: Confidence-based calibration penalty -0.5 * normalized_weight for negatives
    
    Formula:
        final = (1 - penalty_weight) * prefix_score + penalty_weight * calibration_term
        
    Where calibration_term = 0 for positives, -0.5 * normalized_weight for negatives
    (weight ∝ exp(λ * score), normalized within group over negatives only).
    
    Parameters:
        gamma, alpha, num_tokens_per_item: PrefixMatchReward params
        lambda_param: CalibrationReward focusing param
        penalty_weight: Weight for penalty term (0-1), analogous to ndcg_weight
    """
    
    def __init__(
        self,
        num_tokens_per_item: int = 4,
        gamma: float = 0.5,
        alpha: float = 0.5,
        lambda_param: float = 1.0,
        penalty_weight: float = 0.5,
    ):
        """
        Initialize CombinedReward.
        
        Args:
            num_tokens_per_item: T, number of tokens per item
            gamma: Max partial reward upper bound (k=T-1 reaches γ)
            alpha: Shape parameter for prefix. α<1: concave, earlier tokens weighted more
            lambda_param: Focusing param for calibration penalty. λ>1: focus on high-confidence errors
            penalty_weight: Weight for calibration penalty (0-1)
        """
        self.num_tokens_per_item = num_tokens_per_item
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.penalty_weight = penalty_weight
    
    def _compute_prefix_match_score(
        self,
        gen_tokens: List[int],
        target_tokens: List[int],
    ) -> float:
        """Power-law prefix match score. Same logic as PrefixMatchReward."""
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
        
        if matched == 0:
            return 0.0
        if matched >= T:
            return 1.0
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
        generated_scores: Optional[List[float]] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute combined rewards.
        
        Args:
            generated_items: List of generated item IDs
            target_items: List of target item IDs
            num_generations: Number of generations per sample (required)
            generated_tokens: Token sequences for prefix matching
            target_tokens: Target token sequences
            generated_scores: Beam search sequence scores for calibration penalty
        """
        if num_generations is None:
            raise ValueError("num_generations is required for CombinedReward")
        
        if generated_tokens is None or target_tokens is None:
            raise ValueError("generated_tokens and target_tokens are required for CombinedReward")
        
        rewards = []
        
        for group_idx in range(len(generated_items) // num_generations):
            start_idx = group_idx * num_generations
            
            # First pass: prefix scores, pos/neg status, raw weights for negatives
            group_data = []
            for rank in range(num_generations):
                idx = start_idx + rank
                prefix_score = self._compute_prefix_match_score(
                    generated_tokens[idx], target_tokens[idx]
                )
                is_positive = (generated_items[idx] == target_items[idx])
                
                if is_positive:
                    group_data.append((prefix_score, True, None))
                else:
                    if generated_scores is not None and idx < len(generated_scores):
                        score = generated_scores[idx]
                        raw_weight = math.exp(self.lambda_param * score)
                    else:
                        raw_weight = 1.0
                    group_data.append((prefix_score, False, raw_weight))
            
            total_raw = sum(w for _, _, w in group_data if w is not None)
            
            # Second pass: combine prefix + calibration penalty
            for prefix_score, is_positive, raw_weight in group_data:
                if is_positive:
                    cal_term = 0.0
                else:
                    if total_raw > 0:
                        normalized_weight = raw_weight / total_raw
                    else:
                        normalized_weight = 1.0
                    cal_term = -0.5 * normalized_weight
                
                final_reward = (1.0 - self.penalty_weight) * prefix_score + self.penalty_weight * cal_term
                rewards.append(final_reward)
        
        return rewards
