"""
Dual-Weighted Trajectory Aggregation

Implements Section 5.4 of TableMind++.

After plan pruning and confidence-based action refinement we obtain a set
of verified trajectories T* = {(p_i, h_i, y_i)}_{i=1}^{M}.

Each trajectory is assigned a confidence weight:
    w_i = sigma(S_con(p_i)) * C(h_i)

where:
  - sigma(.) is the sigmoid function normalising the unbounded contrastive
    score into (0, 1).
  - C(h_i) is the geometric-mean action confidence of the history h_i.

The final answer is the one that maximises accumulated weight:
    y_hat = argmax_{y in Y}  sum_i I(y_i == y) * w_i
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def _normalize_answer(answer: Any) -> str:
    """
    Convert an answer to a canonical string for comparison.

    Handles common answer types (string, list, number) to ensure
    semantically identical answers are grouped correctly.
    """
    if answer is None:
        return ""
    if isinstance(answer, list):
        # Sort elements for order-independent comparison
        return str(sorted(str(a).strip().lower() for a in answer))
    return str(answer).strip().lower()


class TrajectoryAggregator:
    """
    Synthesises a robust consensus from multiple verified trajectories via
    dual-weighted voting.

    Args:
        fallback_to_majority: If True, falls back to unweighted majority
            voting when all weights are zero (safety net).
    """

    def __init__(self, fallback_to_majority: bool = True):
        self.fallback_to_majority = fallback_to_majority

    def aggregate(
        self,
        trajectories: List[Dict],
    ) -> Tuple[Any, float, Dict[str, float]]:
        """
        Aggregate trajectory results into a final answer.

        Args:
            trajectories: List of dicts, each containing:
              - "plan"             (str):   raw plan text
              - "answer"          (Any):   final derived answer y_i
              - "contrastive_score" (float): S_con(p_i) from PlanPruner
              - "history_confidence" (float): C(h_i) from ActionRefiner

        Returns:
            Tuple (best_answer, best_weight, weight_map) where:
              - best_answer: The answer with the highest accumulated weight.
              - best_weight: The accumulated weight of the best answer.
              - weight_map: Dict mapping each unique answer to its total weight.
        """
        if not trajectories:
            return None, 0.0, {}

        # Compute per-trajectory weights
        weight_map: Dict[str, float] = defaultdict(float)
        answer_map: Dict[str, Any] = {}  # canonical → original answer

        for traj in trajectories:
            s_con = float(traj.get("contrastive_score", 0.0))
            c_hist = float(traj.get("history_confidence", 1.0))
            answer = traj.get("answer")

            # w_i = sigma(S_con(p_i)) * C(h_i)
            weight = _sigmoid(s_con) * max(c_hist, 0.0)

            canonical = _normalize_answer(answer)
            weight_map[canonical] += weight
            if canonical not in answer_map:
                answer_map[canonical] = answer

        if not weight_map:
            return None, 0.0, {}

        # Check if all weights are effectively zero
        total_weight = sum(weight_map.values())
        if total_weight < 1e-12 and self.fallback_to_majority:
            # Fall back to unweighted majority voting
            count_map: Dict[str, int] = defaultdict(int)
            for traj in trajectories:
                canonical = _normalize_answer(traj.get("answer"))
                count_map[canonical] += 1
            best_canonical = max(count_map, key=count_map.__getitem__)
            return answer_map.get(best_canonical), 0.0, dict(weight_map)

        # Select the answer with maximum accumulated weight
        best_canonical = max(weight_map, key=weight_map.__getitem__)
        best_weight = weight_map[best_canonical]
        best_answer = answer_map.get(best_canonical)

        return best_answer, best_weight, dict(weight_map)

    def compute_weight(self, contrastive_score: float, history_confidence: float) -> float:
        """Compute the weight for a single trajectory."""
        return _sigmoid(contrastive_score) * max(history_confidence, 0.0)
