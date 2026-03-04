# genrec/rewards/calibration_reward.py

import math
from typing import List, Optional
from .base_reward import BaseReward


class CalibrationReward(BaseReward):
    """
    Calibration-Aware Reward for Generative Recommendation.
    
    Penalizes negative samples based on their model confidence (beam search probability),
    focusing training on high-confidence errors (hallucinations).
    
    Formula:
        R(y) = 0.5 * I(y == y_gt) - 0.5 * I(y != y_gt) * P_θ(y)^λ
    
    Where:
        - P_θ(y) = exp(sequence_score) is the model's per-token geometric mean confidence
        - λ (lambda_param) is a focusing parameter that controls the penalty concentration
          on high-confidence errors (analogous to γ in Focal Loss)
    
    Comparison with ReRe (GRPOReward):
        - ReRe uses a fixed, rank-based penalty: penalty = f(rank), independent of model state
        - CalibrationReward uses a dynamic, confidence-based penalty: penalty = P_θ(y)^λ,
          which adapts as the model's belief changes during training
    With NDCG-style normalization:
        Negative penalties are normalized within each group so they sum to a fixed value (-0.5),
        ensuring group reward sum = 0 and stable advantage signs (GT positive, negatives negative).
    """
    
    def __init__(self, lambda_param: float = 1.0):
        """
        Initialize CalibrationReward.
        
        Args:
            lambda_param: Focusing parameter (λ ≥ 0). Controls how strongly the penalty
                         concentrates on high-confidence errors.
                         - λ = 0: All negative samples get equal penalty (-0.5)
                         - λ = 1: Penalty proportional to model confidence
                         - λ > 1: Stronger focus on high-confidence errors (recommended 1~3)
                         Analogous to γ in Focal Loss.
        """
        self.lambda_param = lambda_param
    
    def __call__(
        self, 
        generated_items: List[int], 
        target_items: List[int], 
        num_generations: int = None,
        generated_scores: Optional[List[float]] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute calibration-aware rewards.
        
        Args:
            generated_items: List of generated item IDs [B * num_generations]
            target_items: List of target item IDs [B * num_generations]
            num_generations: Number of generations per sample (required)
            generated_scores: List of beam search sequence scores (length-normalized
                            log probabilities) for each generated sequence.
                            Shape: [B * num_generations].
                            For GT sequences, a placeholder value (e.g., 0.0) is expected.
            **kwargs: Unused (for compatibility with other reward kwargs like
                     generated_tokens, target_tokens, etc.)
        
        Returns:
            List of rewards, where:
              - Positive samples (match): 0.5
              - Negative samples (mismatch): -0.5 * w'_i with w'_i = w_i / sum(w_j),
                w_i = exp(λ * score_i), summed over negatives only. Group sum = 0.
        """
        if num_generations is None:
            raise ValueError("num_generations is required for CalibrationReward")
        
        rewards = []
        
        # Process by groups (each group = num_generations candidates for one query)
        for group_idx in range(len(generated_items) // num_generations):
            start_idx = group_idx * num_generations
            
            # First pass: collect positive/negative status and raw weights for negatives
            group_items = []
            for rank in range(num_generations):
                idx = start_idx + rank
                is_match = (generated_items[idx] == target_items[idx])
                if is_match:
                    group_items.append((True, None))
                else:
                    if generated_scores is not None and idx < len(generated_scores):
                        score = generated_scores[idx]
                        raw_weight = math.exp(self.lambda_param * score)
                    else:
                        raw_weight = 1.0
                    group_items.append((False, raw_weight))
            
            # Sum of raw weights over negatives only (for normalization)
            total_raw = sum(w for _, w in group_items if w is not None)
            
            # Second pass: assign rewards (0.5 for positives, -0.5 * normalized_weight for negatives)
            for is_match, raw_weight in group_items:
                if is_match:
                    rewards.append(0.5)
                else:
                    if total_raw > 0:
                        normalized_weight = raw_weight / total_raw
                    else:
                        normalized_weight = 1.0  # fallback when no negatives (should not occur)
                    rewards.append(-0.5 * normalized_weight)
        
        return rewards
