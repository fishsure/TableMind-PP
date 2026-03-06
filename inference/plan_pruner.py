"""
Memory-Guided Plan Pruning

Implements Section 5.2.2 of TableMind++.

Given a set of N candidate plans {p_1, ..., p_N} sampled for query x,
this module:
  1. Encodes x to get E(x) and retrieves top-K historical prototypes.
  2. Partitions retrieved sequences into positive (S_pos) and negative (S_neg).
  3. Computes the contrastive score for each candidate:
         S_con(p_i) = D-(p_i) - D+(p_i)
     where D+(p_i) = min_{A_ref in S_pos} Lev(A(p_i), A_ref)
           D-(p_i) = min_{A_ref in S_neg} Lev(A(p_i), A_ref)
  4. Ranks candidates by S_con and retains the top rho fraction.
"""

from typing import List, Tuple, Optional

import numpy as np

from .memory_builder import MemoryBank, SemanticParser


def _levenshtein(s: List[str], t: List[str]) -> int:
    """
    Compute Levenshtein edit distance between two sequences of strings.

    Each element is treated as a single "character".  The cost of each
    insertion, deletion, or substitution is 1.
    """
    m, n = len(s), len(t)
    # Allocate two rows: previous and current
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def _min_distance(candidate_seq: List[str], reference_seqs: List[List[str]]) -> float:
    """
    Compute D+(p_i) or D-(p_i): the minimum Levenshtein distance from the
    candidate to any reference sequence.

    If reference_seqs is empty, returns a large constant so the candidate
    is not penalised (or rewarded) in the direction with no references.
    """
    if not reference_seqs:
        return float("inf")
    dists = [_levenshtein(candidate_seq, ref) for ref in reference_seqs]
    return float(min(dists))


class PlanPruner:
    """
    Retrieves historical prototypes and prunes candidate plans using the
    contrastive score S_con.

    Args:
        memory_bank: Pre-built :class:`MemoryBank` instance.
        top_k: Number of historical prototypes to retrieve (paper: K=5).
        retention_ratio: Fraction of candidates to retain (paper: rho=0.5).
    """

    def __init__(
        self,
        memory_bank: MemoryBank,
        top_k: int = 5,
        retention_ratio: float = 0.5,
    ):
        self.memory_bank = memory_bank
        self.parser = SemanticParser()
        self.top_k = top_k
        self.retention_ratio = retention_ratio

    def _encode_query(self, query: str) -> np.ndarray:
        return self.memory_bank.encode([query])[0]

    def prune(
        self,
        query: str,
        candidate_plans: List[str],
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Prune candidate plans and return the retained subset with scores.

        Args:
            query: Natural language query for the current instance.
            candidate_plans: List of N plan strings sampled by the model.
            query_embedding: Optional pre-computed query embedding E(x).
                             Computed on-the-fly if not provided.

        Returns:
            Tuple (retained_plans, contrastive_scores) where:
              - retained_plans: Plans in the pruned set P* (top rho fraction).
              - contrastive_scores: S_con values for each retained plan.
        """
        if not candidate_plans:
            return [], []

        # Encode query
        if query_embedding is None:
            query_embedding = self._encode_query(query)

        # Retrieve positive and negative prototypes
        pos_seqs, neg_seqs = self.memory_bank.retrieve(query_embedding, top_k=self.top_k)

        # Parse candidate plans into action sequences
        candidate_seqs = [self.parser.parse(p) for p in candidate_plans]

        # Compute contrastive scores
        scores: List[float] = []
        for seq in candidate_seqs:
            d_pos = _min_distance(seq, pos_seqs)
            d_neg = _min_distance(seq, neg_seqs)

            # Handle edge cases where one side of memory is empty
            if d_pos == float("inf") and d_neg == float("inf"):
                s_con = 0.0
            elif d_pos == float("inf"):
                # No positive references → rely only on negative distance
                s_con = float(d_neg)
            elif d_neg == float("inf"):
                # No negative references → invert positive distance
                s_con = -float(d_pos)
            else:
                s_con = float(d_neg - d_pos)

            scores.append(s_con)

        # Rank by contrastive score (descending) and retain top rho fraction
        n_retain = max(1, int(len(candidate_plans) * self.retention_ratio))
        ranked = sorted(
            zip(candidate_plans, scores), key=lambda x: x[1], reverse=True
        )
        retained_plans = [p for p, _ in ranked[:n_retain]]
        retained_scores = [s for _, s in ranked[:n_retain]]

        return retained_plans, retained_scores

    def score_plan(
        self,
        query: str,
        plan: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the contrastive score for a single plan.

        Useful when scores need to be computed after execution for use in
        dual-weighted trajectory aggregation.
        """
        _, scores = self.prune(query, [plan], query_embedding=query_embedding)
        return scores[0] if scores else 0.0
