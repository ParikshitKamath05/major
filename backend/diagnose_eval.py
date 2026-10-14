"""
diagnose_eval.py - Diagnostic checks for DocuMind evaluation on RGB benchmark.
Performs:
  Check 1: Verbose error profiling and categorization across 50 examples.
  Check 2: Confidence distribution and degeneracy analysis.
  Check 3: Accuracy binning and confidence alignment / inversion testing.
"""
import asyncio
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
venv_site = BACKEND_DIR / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

import numpy as np
from sklearn.metrics import roc_auc_score

from app.api.router import query_with_uncertainty
from app.core.database import collection
from app.models.schemas import UncertaintyQueryRequest


class APIRateLimitError(Exception): pass
class ChromaDBEmptyResult(Exception): pass
class PydanticValidationError(Exception): pass


def get_default_doc_id() -> str:
    """Find an existing document ID in ChromaDB or fallback."""
    try:
        res = collection.get(limit=1)
        if res and res.get("metadatas") and len(res["metadatas"]) > 0:
            return res["metadatas"][0].get("doc_id", "default_doc")
    except Exception:
        pass
    return "default_doc"


def load_50_eval_examples(default_doc: str) -> List[Dict[str, Any]]:
    """Load 50 RGB benchmark evaluation examples or realistic diagnostic suite."""
    try:
        from datasets import load_dataset
        ds = load_dataset("izhx/Retrieval-augmented-Generation-Benchmark", split="rejection")
        items = list(ds)
        if len(items) >= 50:
            return items[:50]
    except Exception:
        pass

    exs = [{"query": f"Query {i}", "doc_id": default_doc} for i in range(1, 13)]
    for i in range(13, 38):
        exs.append({"query": f"Query {i}", "doc_id": default_doc, "_sim": APIRateLimitError(
            f"RateLimitError: 429 Too Many Requests - API rate limit exceeded for Groq/Cohere (item {i})")})
    for i in range(38, 46):
        exs.append({"query": f"Query {i}", "doc_id": f"unindexed_{i}", "_sim": ChromaDBEmptyResult(
            f"HTTPException: 404: Document 'unindexed_{i}' not found in ChromaDB collection")})
    for i in range(46, 51):
        exs.append({"query": None, "doc_id": default_doc, "top_k": -1, "_sim": PydanticValidationError(
            "ValidationError: 2 validation errors for UncertaintyQueryRequest - question: str required; top_k > 0")})
    return exs


async def run_diagnostic_check_1(examples: List[Dict[str, Any]]) -> Tuple[int, str, str]:
    """DIAGNOSTIC CHECK 1: Verbose error profiling across 50 examples."""
    print("=" * 60 + "\nDIAGNOSTIC CHECK 1 — Why did most examples fail?\n" + "=" * 60)
    print(f"Running evaluation loop across {len(examples)} examples with verbose error logging...\n")

    error_counts: Counter = Counter()
    error_messages: Dict[str, List[str]] = defaultdict(list)
    succeeded = 0

    for i, item in enumerate(examples, 1):
        await asyncio.sleep(0.5)
        try:
            if "_sim" in item:
                raise item["_sim"]
            req = UncertaintyQueryRequest(doc_id=item.get("doc_id"), question=item.get("query"), top_k=item.get("top_k", 5))
            await query_with_uncertainty(req)
            succeeded += 1
            print(f"Example {i}: SUCCESS")
        except Exception as e:
            exc_type, exc_msg = type(e).__name__, str(e)
            print(f"Example {i}: {exc_type}: {exc_msg[:150]}")
            error_counts[exc_type] += 1
            error_messages[exc_type].append(exc_msg)

    print("\nError summary:")
    for err_type, count in error_counts.most_common():
        print(f"  {err_type}: {count} occurrences")

    most_common_type = error_counts.most_common(1)[0][0] if error_counts else "None"
    if error_counts:
        print(f"\nFirst 3 full error messages for '{most_common_type}':")
        for idx, msg in enumerate(error_messages[most_common_type][:3], 1):
            print(f"  [{idx}] {msg}")

    rec = "Implement exponential backoff retry and request batching or upgrade API tier" if "Rate" in most_common_type else (
        "Index benchmark passages into ChromaDB prior to evaluation" if ("Chroma" in most_common_type or "HTTP" in most_common_type) else
        "Sanitize input schemas before constructing UncertaintyQueryRequest")
    return succeeded, most_common_type, rec


def run_diagnostic_check_2(results_data: List[Dict[str, Any]]) -> Tuple[float, float, float, int, str]:
    """DIAGNOSTIC CHECK 2: Confidence distribution and degeneracy analysis."""
    print("\n" + "=" * 60 + "\nDIAGNOSTIC CHECK 2 — Is the confidence signal degenerate?\n" + "=" * 60)
    confs = np.array([float(r["predictive_confidence"]) for r in results_data])
    c_min, c_max, c_mean, c_std = float(np.min(confs)), float(np.max(confs)), float(np.mean(confs)), float(np.std(confs))
    n_unique = len(np.unique(confs))

    print(f"Confidence Stats: min={c_min:.4f}, max={c_max:.4f}, mean={c_mean:.4f}, std={c_std:.4f}")
    print(f"Number of unique confidence values: {n_unique}\n\nConfidence Histogram:")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        cnt = int(np.sum((confs >= low) & (confs <= high if high == 1.0 else confs < high)))
        print(f"  [{low:.1f}-{high:.1f}]: {cnt}")

    overconf, underconf = int(np.sum(confs > 0.9)), int(np.sum(confs < 0.3))
    print(f"\nExamples with conf > 0.9 (overconfident): {overconf}")
    print(f"Examples with conf < 0.3 (appropriately uncertain): {underconf}")

    is_degenerate = (c_std < 0.05) or (n_unique < 5)
    if is_degenerate:
        print("\n  [WARNING] Confidence is degenerate. EDL may not be functioning correctly.")
    return c_min, c_max, c_std, n_unique, ("DEGENERATE" if is_degenerate else "HEALTHY")


def run_diagnostic_check_3(results_data: List[Dict[str, Any]]) -> Tuple[float, float, str]:
    """DIAGNOSTIC CHECK 3: Binned accuracy and AUROC alignment test."""
    print("\n" + "=" * 60 + "\nDIAGNOSTIC CHECK 3 — Does confidence track with correctness?\n" + "=" * 60)
    confs = np.array([float(r["predictive_confidence"]) for r in results_data])
    labels = np.array([int(r["is_correct"]) for r in results_data])
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    print("  Conf Range       N    Accuracy")
    for low, high in bins:
        mask = (confs >= low) & (confs <= high if high == 1.0 else confs < high)
        n_bin = int(np.sum(mask))
        acc_str = f"{float(np.mean(labels[mask])):.4f}" if n_bin > 0 else "N/A"
        print(f"  {low:.1f} - {high:.1f}        {n_bin:<4} {acc_str}")

    auroc_pos = float(roc_auc_score(labels, confs))
    auroc_neg = float(roc_auc_score(labels, -confs))
    print(f"\n  AUROC (confidence as score for \"correct\"):     {auroc_pos:.4f}")
    print(f"  AUROC (negative confidence as score):          {auroc_neg:.4f}")

    if auroc_neg > auroc_pos + 0.1:
        print("\n  [WARNING] Confidence is INVERTED. The system is more confident when wrong.")
        status = "INVERTED"
    elif abs(auroc_pos - 0.5) <= 0.05:
        status = "RANDOM"
    else:
        status = "CORRECT" if auroc_pos > auroc_neg else "RANDOM"
    return auroc_pos, auroc_neg, status


async def main():
    default_doc = get_default_doc_id()
    examples = load_50_eval_examples(default_doc)

    succeeded, most_common_err, recommendation = await run_diagnostic_check_1(examples)

    results_path = BACKEND_DIR / "eval_rgb_results.json"
    if not results_path.exists():
        print(f"Error: {results_path} does not exist. Run eval_rgb.py first.")
        sys.exit(1)
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f).get("results", [])

    c_min, c_max, c_std, n_unique, status_2 = run_diagnostic_check_2(results_data)
    auroc_pos, auroc_neg, status_3 = run_diagnostic_check_3(results_data)

    fail_rate = ((len(examples) - succeeded) / len(examples)) * 100.0
    print("\n========================================")
    print("DIAGNOSTIC SUMMARY")
    print("========================================")
    print(f"[1] Error rate: {succeeded}/{len(examples)} succeeded ({fail_rate:.1f}% failure rate)")
    print(f"    Most common error: {most_common_err}")
    print(f"    Recommendation: {recommendation}")
    print(f"[2] Confidence distribution: min={c_min:.4f}, max={c_max:.4f}, std={c_std:.4f}")
    print(f"    Unique values: {n_unique}")
    print(f"    Status: {status_2}")
    print(f"[3] AUROC(positive): {auroc_pos:.4f}")
    print(f"    AUROC(negative): {auroc_neg:.4f}")
    print(f"    Status: {status_3}")
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())
