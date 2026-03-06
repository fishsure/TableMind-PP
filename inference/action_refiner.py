"""
Confidence-Based Action Refinement

Implements Section 5.3 of TableMind++.

To avoid probability dilution by deterministic boilerplate syntax, the
confidence score C(a) is computed *only* over semantically significant
tokens (identifiers, function names, numerical/string literals):

    C(a) = exp( (1/|K|) * sum_{i in K} log P_theta(a_i | x, a_{<i}) )

where K is the index set of key tokens identified via lexical analysis.

If C(a) < tau_code the refinement cycle is triggered: the model is asked
to self-correct the generated code before it is sent to the sandbox.
"""

import re
import tokenize
import io
import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Key-token identification
# ---------------------------------------------------------------------------

# Token types that carry semantic content (not boilerplate syntax)
_KEY_TOKEN_TYPES = {
    tokenize.NAME,    # identifiers and keywords
    tokenize.NUMBER,  # numeric literals
    tokenize.STRING,  # string literals
}

# Python built-in keywords that are deterministic boilerplate
_PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})

# Common pandas / numpy boilerplate identifiers
_BOILERPLATE_NAMES = frozenset({
    "df", "pd", "np", "import", "pandas", "numpy", "print",
    "len", "range", "enumerate", "zip", "list", "dict", "str", "int",
    "float", "bool", "type", "None",
})


def identify_key_tokens(code: str) -> List[str]:
    """
    Use Python's tokenize module to extract semantically significant tokens.

    Key tokens are:
      - Variable identifiers that are not Python keywords or common boilerplate
      - Function/method names (NAME tokens following a def or used as callees)
      - Numeric and string literals

    Returns a list of token strings in the order they appear in the code.
    """
    key_tokens: List[str] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenError:
        # Fallback: simple regex scan for identifiers and literals
        return _regex_key_tokens(code)

    prev_type = None
    for tok_type, tok_str, _, _, _ in tokens:
        if tok_type == tokenize.NAME:
            # Include identifiers that are not keywords or boilerplate
            if tok_str not in _PYTHON_KEYWORDS and tok_str not in _BOILERPLATE_NAMES:
                key_tokens.append(tok_str)
        elif tok_type == tokenize.NUMBER:
            key_tokens.append(tok_str)
        elif tok_type == tokenize.STRING:
            key_tokens.append(tok_str)
        prev_type = tok_type

    return key_tokens


def _regex_key_tokens(code: str) -> List[str]:
    """Fallback regex-based key-token extractor for malformed code snippets."""
    # Match identifiers and numeric literals
    pattern = re.compile(r"\b([A-Za-z_]\w*|\d+(?:\.\d+)?)\b|\"[^\"]*\"|'[^']*'")
    tokens = []
    for m in pattern.finditer(code):
        tok = m.group(0)
        # Exclude Python keywords and boilerplate
        if tok in _PYTHON_KEYWORDS or tok in _BOILERPLATE_NAMES:
            continue
        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def compute_key_token_confidence(
    code: str,
    token_logprobs: List[Dict],
) -> float:
    """
    Compute C(a) = exp( mean_{i in K} log P(a_i) ) over key-token positions.

    Args:
        code: The generated code string.
        token_logprobs: List of dicts from the vLLM/OpenAI logprobs response.
            Each dict must contain at minimum:
              - "token"   (str): the token string
              - "logprob" (float): log probability of this token

    Returns:
        Confidence score in (0, 1].  Returns 1.0 if no key tokens are found
        (conservative: do not trigger refinement when we cannot measure).
    """
    if not token_logprobs:
        return 1.0

    key_token_set = set(identify_key_tokens(code))
    if not key_token_set:
        return 1.0

    selected_logprobs: List[float] = []
    for entry in token_logprobs:
        tok = entry.get("token", "").strip()
        lp = entry.get("logprob", 0.0)
        if tok in key_token_set:
            selected_logprobs.append(lp)

    if not selected_logprobs:
        return 1.0

    mean_logprob = sum(selected_logprobs) / len(selected_logprobs)
    confidence = math.exp(mean_logprob)
    return float(confidence)


def compute_history_confidence(action_confidences: List[float]) -> float:
    """
    Compute C(h_i) for a full trajectory history h_i.

    The confidence of the history is the product of per-action confidences,
    which equals the exponential of the sum of log-confidences.
    We take the geometric mean across actions for numerical stability.

    Args:
        action_confidences: List of C(a) scores for each action in h_i.

    Returns:
        C(h_i) in (0, 1].
    """
    if not action_confidences:
        return 1.0
    log_conf = sum(math.log(max(c, 1e-12)) for c in action_confidences)
    return math.exp(log_conf / len(action_confidences))


# ---------------------------------------------------------------------------
# Action refiner
# ---------------------------------------------------------------------------

class ActionRefiner:
    """
    Confidence-Based Action Refinement.

    Monitors the token-level confidence of generated code actions and
    triggers a self-correction request when confidence falls below tau.

    Args:
        confidence_threshold: tau in the paper (default 0.8 per Table 6).
        max_refinement_attempts: Maximum number of self-correction cycles.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        max_refinement_attempts: int = 2,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_refinement_attempts = max_refinement_attempts

    def needs_refinement(self, confidence: float) -> bool:
        """Return True if the confidence score is below the threshold."""
        return confidence < self.confidence_threshold

    def build_refinement_prompt(self, original_code: str, low_conf_tokens: List[str]) -> str:
        """
        Build a prompt asking the model to self-correct uncertain code.

        Args:
            original_code: The generated code that triggered refinement.
            low_conf_tokens: The key tokens with low generation confidence.

        Returns:
            Instruction string to be prepended to the conversation.
        """
        token_hint = ", ".join(f'"{t}"' for t in low_conf_tokens[:10])
        return (
            "The previous code action may contain errors. "
            f"Please review and correct it, paying special attention to: {token_hint}. "
            "Rewrite the corrected code.\n\n"
            f"Original code:\n```python\n{original_code}\n```\n\n"
            "Corrected code:"
        )

    def get_low_confidence_tokens(
        self,
        code: str,
        token_logprobs: List[Dict],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Return key tokens whose log-probability is below an individual threshold.

        Used to provide targeted hints in the refinement prompt.
        """
        if threshold is None:
            threshold = self.confidence_threshold
        key_set = set(identify_key_tokens(code))
        low_conf = []
        for entry in token_logprobs:
            tok = entry.get("token", "").strip()
            lp = entry.get("logprob", 0.0)
            if tok in key_set and math.exp(lp) < threshold:
                low_conf.append(tok)
        return low_conf
