"""
TableMind++ — Uncertainty-Aware Inference Orchestrator

End-to-end implementation of the TableMind++ inference pipeline (Section 5).

Pipeline overview:
  1. Generate N candidate plans with temperature sampling (default N=16).
  2. Memory-Guided Plan Pruning: encode query → retrieve K=5 prototypes →
     compute contrastive scores → retain top rho=0.5 candidates.
  3. For each retained plan, run the multi-turn plan-action-reflect loop
     with Confidence-Based Action Refinement (token-level confidence < tau
     triggers self-correction before sandbox execution).
  4. Dual-Weighted Trajectory Aggregation: synthesise the final answer
     via sigma(S_con) * C(h_i) weighted voting.

The agent communicates with a vLLM server through the OpenAI-compatible
API.  Logprobs (``logprobs=True``) are requested for every action
generation call to compute key-token confidence scores.
"""

import json
import re
import math
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .memory_builder import MemoryBank, SemanticParser
from .plan_pruner import PlanPruner
from .action_refiner import ActionRefiner, compute_key_token_confidence, compute_history_confidence
from .trajectory_aggregator import TrajectoryAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_PLAN_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a reasoning agent that solves problems by thinking and using tools. "
    "Follow the plan-action-reflect loop:\n"
    "1. Formulate a step-by-step plan inside <think>...</think>.\n"
    "2. Write a Python code action inside <tool_call>{\"name\": \"python\", "
    "\"arguments\": {\"code\": \"...\", \"files\": [...]}}</tool_call>.\n"
    "3. Observe the execution result and reflect.\n"
    "4. When finished, provide the final answer inside <answer>...</answer>."
)


def _extract_answer(text: str) -> Optional[str]:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_plan(text: str) -> str:
    """Extract the plan from the first <think> block."""
    m = _PLAN_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_tool_call(text: str) -> Optional[Dict]:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _format_table_prompt(table_text: str, question: str) -> str:
    return (
        f"## Table Content:\n{table_text}\n\n"
        f"## Question: {question}\n"
    )


def _get_token_logprobs(choice) -> List[Dict]:
    """
    Extract token logprobs from an OpenAI API response choice object.

    Returns a list of {"token": str, "logprob": float} dicts.
    """
    logprobs_obj = getattr(choice, "logprobs", None)
    if logprobs_obj is None:
        return []
    content = getattr(logprobs_obj, "content", None)
    if content is None:
        return []
    result = []
    for item in content:
        tok = getattr(item, "token", "")
        lp = getattr(item, "logprob", 0.0)
        result.append({"token": tok, "logprob": lp})
    return result


# ---------------------------------------------------------------------------
# Sandbox execution (delegates to vLLM tool environment)
# ---------------------------------------------------------------------------

def _execute_code_via_sandbox(code: str, files: List[str]) -> Tuple[str, bool]:
    """
    Execute Python code in the sandbox.

    Returns (output, success).  Requires sandbox_fusion to be running.
    """
    try:
        from sandbox_fusion import run_code, RunCodeRequest, RunStatus
        import base64
        import os

        req = RunCodeRequest(code=code, language="python", run_timeout=10)
        result = run_code(request=req)
        if result.status == RunStatus.Success:
            stdout = (result.run_result.stdout or "").strip() or "Execution successful but no output"
            return stdout, True
        else:
            err = ""
            if result.run_result and result.run_result.stderr:
                err = result.run_result.stderr.strip().splitlines()[-1]
            elif result.compile_result and result.compile_result.stderr:
                err = result.compile_result.stderr
            return err or "Execution failed", False
    except Exception as exc:
        return f"Sandbox error: {exc}", False


# ---------------------------------------------------------------------------
# TableMind++ Inference
# ---------------------------------------------------------------------------

class TableMindPP:
    """
    Uncertainty-Aware Programmatic Agent for Tool-Augmented Table Reasoning.

    Args:
        memory_bank_path: Path to a saved :class:`~inference.memory_builder.MemoryBank`.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key (default ``"EMPTY"`` for local vLLM).
        model_name: Model name served by vLLM.
        num_candidates: N — number of candidate plans to sample (default 16).
        top_k_memory: K — number of prototypes to retrieve from memory (default 5).
        retention_ratio: rho — fraction of plans to retain after pruning (default 0.5).
        confidence_threshold: tau — token confidence threshold for action refinement (default 0.8).
        max_turns: Maximum number of plan-action-reflect turns per trajectory.
        temperature: Sampling temperature for plan generation.
        max_tokens: Maximum tokens per generation call.
    """

    def __init__(
        self,
        memory_bank_path: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "tablemind",
        num_candidates: int = 16,
        top_k_memory: int = 5,
        retention_ratio: float = 0.5,
        confidence_threshold: float = 0.8,
        max_turns: int = 3,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ):
        self.num_candidates = num_candidates
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

        # Load memory bank
        self.memory_bank = MemoryBank.load(memory_bank_path)

        # Initialise components
        self.plan_pruner = PlanPruner(
            memory_bank=self.memory_bank,
            top_k=top_k_memory,
            retention_ratio=retention_ratio,
        )
        self.action_refiner = ActionRefiner(
            confidence_threshold=confidence_threshold,
        )
        self.aggregator = TrajectoryAggregator()
        self.parser = SemanticParser()

        # vLLM client
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    # ------------------------------------------------------------------
    # Core generation helpers
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[Dict],
        n: int = 1,
        temperature: Optional[float] = None,
        logprobs: bool = False,
    ):
        """Call the vLLM API and return the list of choices."""
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
            n=n,
            logprobs=logprobs,
            top_logprobs=1 if logprobs else None,
        )
        response = self.client.chat.completions.create(**kwargs)
        return response.choices

    def _sample_plans(self, user_message: str) -> Tuple[List[str], List[str]]:
        """
        Sample N candidate first-turn responses and extract (full_response, plan).

        Returns:
            Tuple (full_responses, plans) — both lists of length N (or fewer if
            the model produced fewer outputs).
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        choices = self._generate(messages, n=self.num_candidates, temperature=self.temperature)
        full_responses = [c.message.content or "" for c in choices]
        plans = [_extract_plan(r) for r in full_responses]
        return full_responses, plans

    def _execute_trajectory(
        self,
        user_message: str,
        first_response: str,
        contrastive_score: float,
    ) -> Dict:
        """
        Run the full multi-turn plan-action-reflect loop for one trajectory.

        Applies confidence-based action refinement at each code generation step.

        Args:
            user_message: The full user prompt (table + question).
            first_response: The model's first-turn response (plan + first action).
            contrastive_score: S_con(p_i) from plan pruning.

        Returns:
            Trajectory dict with keys: plan, answer, contrastive_score,
            history_confidence, success.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": first_response},
        ]

        action_confidences: List[float] = []
        plan_text = _extract_plan(first_response)
        answer = _extract_answer(first_response)

        for turn in range(self.max_turns):
            # Check if the latest assistant message contains a tool call
            last_assistant = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
            tool_call_data = _extract_tool_call(last_assistant)

            if tool_call_data is None:
                # No tool call → the model may have produced the final answer
                if answer is None:
                    answer = _extract_answer(last_assistant)
                break

            # Extract code from tool call arguments
            args = tool_call_data.get("arguments", {})
            code = args.get("code", "")
            files = args.get("files", [])

            # --- Action refinement: check confidence of generated code ---
            # Re-generate with logprobs to compute confidence
            temp_messages = messages[:-1]  # exclude the latest assistant turn
            refinement_choices = self._generate(
                temp_messages, n=1, temperature=self.temperature, logprobs=True
            )
            if refinement_choices:
                token_logprobs = _get_token_logprobs(refinement_choices[0])
                confidence = compute_key_token_confidence(code, token_logprobs)
            else:
                confidence = 1.0

            action_confidences.append(confidence)

            # Trigger refinement if below threshold
            for _attempt in range(self.action_refiner.max_refinement_attempts):
                if not self.action_refiner.needs_refinement(confidence):
                    break
                # Build refinement prompt
                low_conf_tokens = self.action_refiner.get_low_confidence_tokens(
                    code, token_logprobs if refinement_choices else []
                )
                refinement_prompt = self.action_refiner.build_refinement_prompt(
                    code, low_conf_tokens
                )
                # Ask the model to self-correct
                refine_messages = temp_messages + [
                    {"role": "user", "content": refinement_prompt},
                ]
                refine_choices = self._generate(
                    refine_messages, n=1, temperature=0.3, logprobs=True
                )
                if refine_choices:
                    refined_text = refine_choices[0].message.content or ""
                    # Extract corrected code (look for code block)
                    code_match = re.search(r"```python\n(.*?)```", refined_text, re.DOTALL)
                    if code_match:
                        code = code_match.group(1).strip()
                        token_logprobs = _get_token_logprobs(refine_choices[0])
                        confidence = compute_key_token_confidence(code, token_logprobs)
                        action_confidences[-1] = confidence  # update last entry
            # Execute code in sandbox
            output, success = _execute_code_via_sandbox(code, files)

            # Append tool response to conversation
            tool_response = (
                f"<tool_response>\n{output[:2048]}\n</tool_response>"
            )
            messages.append({"role": "user", "content": tool_response})

            # Generate next assistant turn
            next_choices = self._generate(messages, n=1, temperature=self.temperature)
            if not next_choices:
                break
            next_content = next_choices[0].message.content or ""
            messages.append({"role": "assistant", "content": next_content})

            answer = _extract_answer(next_content)
            if answer is not None:
                break

        history_confidence = compute_history_confidence(action_confidences)

        return {
            "plan": plan_text,
            "answer": answer,
            "contrastive_score": contrastive_score,
            "history_confidence": history_confidence,
            "success": answer is not None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(
        self,
        question: str,
        table_text: str,
        return_details: bool = False,
    ) -> Any:
        """
        Run the full TableMind++ inference pipeline.

        Args:
            question: Natural language question about the table.
            table_text: Serialised table (markdown / CSV / plain text).
            return_details: If True, also return the intermediate trajectory
                            dicts and weight map.

        Returns:
            The predicted answer string (or None if no answer was derived).
            If ``return_details=True``, returns (answer, trajectories, weight_map).
        """
        user_message = _format_table_prompt(table_text, question)

        # Step 1: Sample N candidate first-turn responses
        print(f"[TableMind++] Sampling {self.num_candidates} candidate plans...")
        first_responses, plans = self._sample_plans(user_message)

        # Step 2: Memory-guided plan pruning
        query_embedding = self.memory_bank.encode([question])[0]
        retained_plans, contrastive_scores = self.plan_pruner.prune(
            query=question,
            candidate_plans=plans,
            query_embedding=query_embedding,
        )

        # Map retained plans back to their first responses
        plan_to_response: Dict[str, str] = {}
        for resp, plan in zip(first_responses, plans):
            if plan not in plan_to_response:
                plan_to_response[plan] = resp

        print(
            f"[TableMind++] Retained {len(retained_plans)} plans "
            f"(from {len(plans)}) after pruning."
        )

        # Step 3: Execute each retained trajectory with action refinement
        trajectories = []
        for plan, s_con in zip(retained_plans, contrastive_scores):
            first_resp = plan_to_response.get(plan, "")
            traj = self._execute_trajectory(
                user_message=user_message,
                first_response=first_resp,
                contrastive_score=s_con,
            )
            trajectories.append(traj)
            status = "OK" if traj["success"] else "NO ANSWER"
            print(
                f"[TableMind++]  Trajectory answer={traj['answer']!r} "
                f"conf={traj['history_confidence']:.3f} [{status}]"
            )

        # Step 4: Dual-weighted trajectory aggregation
        best_answer, best_weight, weight_map = self.aggregator.aggregate(trajectories)
        print(f"[TableMind++] Final answer: {best_answer!r} (weight={best_weight:.4f})")

        if return_details:
            return best_answer, trajectories, weight_map
        return best_answer
