"""
Memory-Guided Plan Pruning: Dual-Memory Bank Construction

Implements the offline memory bank construction described in TableMind++ Section 5.2.1.
Each memory entry is stored as (x_j, p_j, E(x_j), A(p_j)) where:
  - x_j: raw natural language query
  - p_j: raw plan text
  - E(x_j): dense vector embedding (retrieval key)
  - A(p_j): abstracted action sequence (structural template)

Memory is categorized into M+ (positive, correct trajectories)
and M- (negative, deceptive but executable trajectories).
"""

import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Action primitives and keyword mappings (O in the paper)
# ---------------------------------------------------------------------------

ACTION_PRIMITIVES = [
    "FILTER",
    "GROUP",
    "AGGREGATE",
    "SORT",
    "JOIN",
    "COMPUTE",
    "SELECT",
    "MERGE",
    "PIVOT",
    "RENAME",
]

# Keyword → primitive mapping
KEYWORD_MAP: Dict[str, str] = {
    # FILTER
    "filter": "FILTER",
    "remove rows": "FILTER",
    "keep only": "FILTER",
    "select rows": "FILTER",
    "where": "FILTER",
    "condition": "FILTER",
    "drop rows": "FILTER",
    "exclude": "FILTER",
    "mask": "FILTER",
    "subset": "FILTER",
    # GROUP
    "group": "GROUP",
    "groupby": "GROUP",
    "group by": "GROUP",
    "categorize": "GROUP",
    "partition": "GROUP",
    "segment": "GROUP",
    "cluster": "GROUP",
    # AGGREGATE
    "aggregate": "AGGREGATE",
    "sum": "AGGREGATE",
    "average": "AGGREGATE",
    "mean": "AGGREGATE",
    "count": "AGGREGATE",
    "total": "AGGREGATE",
    "maximum": "AGGREGATE",
    "minimum": "AGGREGATE",
    "max": "AGGREGATE",
    "min": "AGGREGATE",
    "median": "AGGREGATE",
    "std": "AGGREGATE",
    "variance": "AGGREGATE",
    # SORT
    "sort": "SORT",
    "order": "SORT",
    "rank": "SORT",
    "arrange": "SORT",
    "ascending": "SORT",
    "descending": "SORT",
    # JOIN
    "join": "JOIN",
    "merge": "MERGE",
    "combine": "JOIN",
    "concat": "JOIN",
    "concatenate": "JOIN",
    "append": "JOIN",
    "link": "JOIN",
    # COMPUTE
    "calculate": "COMPUTE",
    "compute": "COMPUTE",
    "subtract": "COMPUTE",
    "divide": "COMPUTE",
    "multiply": "COMPUTE",
    "add": "COMPUTE",
    "difference": "COMPUTE",
    "ratio": "COMPUTE",
    "percentage": "COMPUTE",
    "convert": "COMPUTE",
    # SELECT
    "select": "SELECT",
    "extract": "SELECT",
    "retrieve": "SELECT",
    "locate": "SELECT",
    "find": "SELECT",
    "get": "SELECT",
    "look up": "SELECT",
    # PIVOT
    "pivot": "PIVOT",
    "reshape": "PIVOT",
    "transpose": "PIVOT",
    "melt": "PIVOT",
    "unstack": "PIVOT",
    # RENAME
    "rename": "RENAME",
    "relabel": "RENAME",
}


class SemanticParser:
    """
    Canonicalizes free-form planning text into a sequence of logical primitives.

    The parser identifies reasoning keywords and maps them to primitives from the
    predefined set O = {FILTER, GROUP, AGGREGATE, SORT, JOIN, COMPUTE, SELECT,
    MERGE, PIVOT, RENAME}, discarding schema-specific arguments (column names,
    numerical literals) to capture pure logical structure.
    """

    def __init__(self):
        # Build a sorted list of (keyword, primitive) tuples, longest first to
        # ensure maximal-munch matching.
        self._patterns: List[Tuple[re.Pattern, str]] = []
        for kw, prim in sorted(KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
            # Word-boundary aware, case-insensitive
            pat = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            self._patterns.append((pat, prim))

    def parse(self, plan_text: str) -> List[str]:
        """
        Parse a plan text and return the abstracted action sequence.

        Args:
            plan_text: Free-form natural language plan.

        Returns:
            Ordered list of primitive operation strings, e.g.
            ["FILTER", "SORT", "AGGREGATE"].
        """
        if not plan_text:
            return []

        # Find all matches with their positions
        hits: List[Tuple[int, str]] = []
        for pat, prim in self._patterns:
            for m in pat.finditer(plan_text):
                hits.append((m.start(), prim))

        # Sort by position and deduplicate consecutive identical primitives
        hits.sort(key=lambda x: x[0])
        sequence: List[str] = []
        for _, prim in hits:
            if not sequence or sequence[-1] != prim:
                sequence.append(prim)

        return sequence


# ---------------------------------------------------------------------------
# Dual Memory Bank
# ---------------------------------------------------------------------------


class MemoryBank:
    """
    Dual-memory bank storing positive (M+) and negative (M-) reasoning trajectories.

    Entries are stored as dicts with fields:
      - query (str): raw natural language query
      - plan (str): raw plan text
      - embedding (np.ndarray): dense vector of the query
      - action_sequence (List[str]): abstracted action sequence A(p)
      - is_positive (bool): True → M+, False → M-

    Usage (offline construction)::

        bank = MemoryBank(encoder_name="BAAI/bge-m3")
        bank.build_from_trajectories(trajectories)
        bank.save("memory_bank.pkl")

    Usage (online inference)::

        bank = MemoryBank.load("memory_bank.pkl")
        pos_seqs, neg_seqs = bank.retrieve(query_embedding, top_k=5)
    """

    def __init__(self, encoder_name: str = "BAAI/bge-m3"):
        self.encoder_name = encoder_name
        self._encoder = None  # lazy-load

        self.parser = SemanticParser()
        self._entries: List[Dict] = []

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for MemoryBank. "
                    "Install with: pip install sentence-transformers"
                ) from e
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into dense vectors."""
        enc = self._get_encoder()
        embeddings = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_from_trajectories(
        self,
        trajectories: List[Dict],
        batch_size: int = 64,
    ) -> None:
        """
        Populate the memory bank from a list of trajectory dicts.

        Each trajectory dict must have:
          - "query" (str)
          - "plan" (str)
          - "is_correct" (bool): True → M+, False → M-

        Args:
            trajectories: List of trajectory dicts.
            batch_size: Batch size for embedding computation.
        """
        queries = [t["query"] for t in trajectories]
        # Encode in batches
        enc = self._get_encoder()
        embeddings = enc.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self._entries = []
        for t, emb in zip(trajectories, embeddings):
            action_seq = self.parser.parse(t["plan"])
            self._entries.append(
                {
                    "query": t["query"],
                    "plan": t["plan"],
                    "embedding": emb.astype(np.float32),
                    "action_sequence": action_seq,
                    "is_positive": bool(t["is_correct"]),
                }
            )

    def add_entry(
        self,
        query: str,
        plan: str,
        is_correct: bool,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Add a single entry to the memory bank."""
        if embedding is None:
            embedding = self.encode([query])[0]
        action_seq = self.parser.parse(plan)
        self._entries.append(
            {
                "query": query,
                "plan": plan,
                "embedding": embedding.astype(np.float32),
                "action_sequence": action_seq,
                "is_positive": is_correct,
            }
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Retrieve top-K nearest neighbors from M+ and M- separately.

        Args:
            query_embedding: Dense vector of the current query (shape [D]).
            top_k: Number of prototypes to retrieve from each sub-bank.

        Returns:
            Tuple (pos_sequences, neg_sequences) where each is a list of
            action sequence lists from M+ and M- respectively.
        """
        if not self._entries:
            return [], []

        pos_entries = [e for e in self._entries if e["is_positive"]]
        neg_entries = [e for e in self._entries if not e["is_positive"]]

        def top_k_seqs(entries: List[Dict], k: int) -> List[List[str]]:
            if not entries:
                return []
            embs = np.stack([e["embedding"] for e in entries])
            sims = embs @ query_embedding  # cosine similarity (normalized)
            k = min(k, len(entries))
            idx = np.argpartition(sims, -k)[-k:]
            return [entries[i]["action_sequence"] for i in idx]

        pos_seqs = top_k_seqs(pos_entries, top_k)
        neg_seqs = top_k_seqs(neg_entries, top_k)
        return pos_seqs, neg_seqs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the memory bank to a pickle file (encoder not saved)."""
        data = {
            "encoder_name": self.encoder_name,
            "entries": self._entries,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[MemoryBank] Saved {len(self._entries)} entries to {path}")

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        """Load a previously saved memory bank."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        bank = cls(encoder_name=data["encoder_name"])
        bank._entries = data["entries"]
        pos = sum(1 for e in bank._entries if e["is_positive"])
        neg = len(bank._entries) - pos
        print(
            f"[MemoryBank] Loaded {len(bank._entries)} entries "
            f"(M+={pos}, M-={neg}) from {path}"
        )
        return bank

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def num_positive(self) -> int:
        return sum(1 for e in self._entries if e["is_positive"])

    @property
    def num_negative(self) -> int:
        return sum(1 for e in self._entries if not e["is_positive"])
