"""
Offline Memory Bank Construction Script

Usage:
    python scripts/build_memory.py \
        --model-path /path/to/trained/tablemind \
        --train-data data/train.parquet \
        --output memory_bank.pkl \
        [--encoder BAAI/bge-m3] \
        [--batch-size 64] \
        [--api-base http://localhost:8000/v1]

This script:
  1. Runs the trained TableMind model on all training instances to collect
     self-generated reasoning trajectories (positive: correct, negative:
     executable but wrong).
  2. Encodes each query with the sentence encoder.
  3. Saves the dual-memory bank to disk.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI


def parse_args():
    p = argparse.ArgumentParser(description="Build TableMind++ dual-memory bank")
    p.add_argument("--model-path", type=str, required=True, help="Trained model path or name")
    p.add_argument("--train-data", type=str, required=True, help="Training parquet file")
    p.add_argument("--output", type=str, default="memory_bank.pkl", help="Output memory bank path")
    p.add_argument("--encoder", type=str, default="BAAI/bge-m3", help="Sentence encoder model")
    p.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    p.add_argument("--api-key", type=str, default="EMPTY")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0, help="Use 0 for greedy decoding")
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64, help="Encoder batch size")
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples")
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are a reasoning agent that solves problems by thinking and using tools. "
    "Follow the plan-action-reflect loop with <think>, <tool_call>, and <answer> tags."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_PLAN_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def extract_answer(text: str):
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def extract_plan(text: str) -> str:
    m = _PLAN_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def normalize_answer(answer):
    if answer is None:
        return ""
    return str(answer).strip().lower()


def check_correct(predicted, ground_truth):
    if predicted is None:
        return False
    pred_norm = normalize_answer(predicted)
    if isinstance(ground_truth, list):
        return any(pred_norm == normalize_answer(g) for g in ground_truth)
    return pred_norm == normalize_answer(ground_truth)


def run_single_inference(client, model_name, table_text, question, max_tokens, temperature):
    """Run single-pass greedy inference and return (plan, answer, full_response)."""
    user_msg = f"## Table Content:\n{table_text}\n\n## Question: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        content = response.choices[0].message.content or ""
        plan = extract_plan(content)
        answer = extract_answer(content)
        return plan, answer, content
    except Exception as exc:
        print(f"  [WARNING] Inference failed: {exc}")
        return "", None, ""


def build_memory(args):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference.memory_builder import MemoryBank

    # Load training data
    print(f"Loading training data from {args.train_data} ...")
    df = pd.read_parquet(args.train_data)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"  {len(df)} training instances loaded.")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    trajectories = []
    for idx, row in df.iterrows():
        # Extract table and question from the parquet row
        prompt = row.get("prompt", [])
        if isinstance(prompt, list):
            # Chat format: find the user message
            user_content = next(
                (m["content"] for m in prompt if m.get("role") == "user"), ""
            )
        else:
            user_content = str(prompt)

        # Ground truth
        ground_truth = None
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")

        # Extract table and question from user content
        table_match = re.search(r"## Table Content:\n(.*?)\n\n## Question:", user_content, re.DOTALL)
        q_match = re.search(r"## Question:\s*(.*)", user_content, re.DOTALL)
        table_text = table_match.group(1).strip() if table_match else user_content
        question = q_match.group(1).strip() if q_match else user_content

        # Run inference to generate trajectory
        plan, predicted, _ = run_single_inference(
            client, args.model_path, table_text, question,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )

        is_correct = check_correct(predicted, ground_truth)

        # Include only trajectories that produced an answer (positive or negative)
        # Skip trajectories where the model produced no answer at all
        if predicted is None and not is_correct:
            continue

        trajectories.append({
            "query": question,
            "plan": plan,
            "is_correct": is_correct,
        })

        if (idx + 1) % 100 == 0:
            pos = sum(1 for t in trajectories if t["is_correct"])
            neg = len(trajectories) - pos
            print(f"  [{idx + 1}/{len(df)}] Collected M+={pos}, M-={neg}")

    pos = sum(1 for t in trajectories if t["is_correct"])
    neg = len(trajectories) - pos
    print(f"\nTotal trajectories: {len(trajectories)} (M+={pos}, M-={neg})")

    if not trajectories:
        print("[ERROR] No trajectories collected. Is the vLLM server running?")
        return

    # Build and save memory bank
    print(f"\nBuilding memory bank with encoder '{args.encoder}' ...")
    bank = MemoryBank(encoder_name=args.encoder)
    bank.build_from_trajectories(trajectories, batch_size=args.batch_size)
    bank.save(args.output)
    print(f"Memory bank saved to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    build_memory(args)
