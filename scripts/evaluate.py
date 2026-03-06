"""
TableMind++ Evaluation Script

Evaluates the TableMind++ model on standard table reasoning benchmarks.

Usage:
    python scripts/evaluate.py \
        --data-path data/test.parquet \
        --memory-bank memory_bank.pkl \
        --dataset WTQ \
        [--api-base http://localhost:8000/v1] \
        [--model tablemind] \
        [--num-candidates 16] \
        [--output-path results.json]

Supported datasets: WTQ, TabMWP, TabFact, HiTab, FinQA
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TableMind++ on benchmark datasets")
    p.add_argument("--data-path", type=str, required=True, help="Test parquet/jsonl file")
    p.add_argument("--memory-bank", type=str, required=True, help="Path to memory_bank.pkl")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["WTQ", "TabMWP", "TabFact", "HiTab", "FinQA"],
                   help="Dataset name for metric selection")
    p.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    p.add_argument("--api-key", type=str, default="EMPTY")
    p.add_argument("--model", type=str, default="tablemind")
    p.add_argument("--num-candidates", type=int, default=16)
    p.add_argument("--top-k-memory", type=int, default=5)
    p.add_argument("--retention-ratio", type=float, default=0.5)
    p.add_argument("--confidence-threshold", type=float, default=0.8)
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-path", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def normalize_number(s: str) -> float:
    """Try to parse a string as a number."""
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return float("nan")


def exact_match(pred, gold) -> bool:
    if pred is None:
        return False
    pred_s = str(pred).strip().lower()
    if isinstance(gold, list):
        return any(pred_s == str(g).strip().lower() for g in gold)
    return pred_s == str(gold).strip().lower()


def numeric_match(pred, gold, tol: float = 1e-2) -> bool:
    """Match with tolerance for numeric answers."""
    if pred is None:
        return False
    pn = normalize_number(str(pred))
    if isinstance(gold, list):
        gns = [normalize_number(str(g)) for g in gold]
    else:
        gns = [normalize_number(str(gold))]
    import math
    return any(not math.isnan(pn) and not math.isnan(gn) and abs(pn - gn) < tol for gn in gns)


def fact_match(pred, gold) -> bool:
    """TabFact: entailed / refuted binary classification."""
    if pred is None:
        return False
    pred_s = str(pred).strip().lower()
    # Accept "1"/"true"/"entailed" → "entailed", "0"/"false"/"refuted" → "refuted"
    if pred_s in ("1", "true", "yes", "entailed"):
        pred_s = "entailed"
    elif pred_s in ("0", "false", "no", "refuted"):
        pred_s = "refuted"
    if isinstance(gold, list):
        return any(pred_s == str(g).strip().lower() for g in gold)
    return pred_s == str(gold).strip().lower()


def compute_metric(dataset: str, pred, gold) -> float:
    if dataset in ("WTQ", "HiTab"):
        return float(exact_match(pred, gold) or numeric_match(pred, gold))
    elif dataset in ("TabMWP", "FinQA"):
        return float(numeric_match(pred, gold) or exact_match(pred, gold))
    elif dataset == "TabFact":
        return float(fact_match(pred, gold))
    return float(exact_match(pred, gold))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> list:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    elif path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    elif path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def extract_table_and_question(row: dict):
    """Extract table text and question from a row dict."""
    # Try direct fields
    if "table" in row and "question" in row:
        return str(row["table"]), str(row["question"])
    # Try from prompt (chat format)
    prompt = row.get("prompt", [])
    if isinstance(prompt, list):
        user_msg = next((m["content"] for m in prompt if m.get("role") == "user"), "")
    else:
        user_msg = str(prompt)
    table_m = re.search(r"## Table Content:\n(.*?)\n\n## Question:", user_msg, re.DOTALL)
    q_m = re.search(r"## Question:\s*(.*)", user_msg, re.DOTALL)
    table = table_m.group(1).strip() if table_m else user_msg
    question = q_m.group(1).strip() if q_m else user_msg
    return table, question


def extract_ground_truth(row: dict):
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    return row.get("answer") or row.get("ground_truth")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    from inference.tablemind_pp import TableMindPP

    agent = TableMindPP(
        memory_bank_path=args.memory_bank,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        num_candidates=args.num_candidates,
        top_k_memory=args.top_k_memory,
        retention_ratio=args.retention_ratio,
        confidence_threshold=args.confidence_threshold,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    records = load_data(args.data_path)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"Evaluating on {len(records)} instances from {args.dataset}...")

    results = []
    total_score = 0.0

    for i, row in enumerate(records):
        table_text, question = extract_table_and_question(row)
        ground_truth = extract_ground_truth(row)

        predicted = agent.infer(question=question, table_text=table_text)
        score = compute_metric(args.dataset, predicted, ground_truth)
        total_score += score

        results.append(
            {
                "index": i,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "score": score,
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            acc = total_score / (i + 1) * 100
            print(f"  [{i + 1}/{len(records)}] Running accuracy: {acc:.2f}%")

    final_acc = total_score / len(records) * 100 if records else 0.0
    summary = {
        "dataset": args.dataset,
        "num_instances": len(records),
        "accuracy": final_acc,
        "results": results,
    }
    print(f"\n{'=' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy: {final_acc:.2f}% ({int(total_score)}/{len(records)})")
    print(f"{'=' * 60}")

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_path}")

    return final_acc


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
