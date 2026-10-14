"""
eval_rgb.py - DocuMind evaluation on SQuAD 2.0 (150 balanced examples).

Changes from original:
  - 75 answerable + 75 unanswerable, seed=42
  - 6.0 s sleep between queries (~10 req/min, safe under Cohere trial)
  - Exponential-backoff retry (5 s / 10 s / 20 s) ONLY on rate-limit errors
  - Non-retryable errors: log + skip; NEVER substitute synthetic results
  - Timestamped output: eval_results_squad2_<ts>.json
  - Abstention-analysis table at confidence thresholds 0.3/0.4/0.5/0.6/0.7
  - AUROC reported both as positive and negative confidence
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ──────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
venv_site = BACKEND_DIR / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

import numpy as np
from dotenv import load_dotenv
from netcal.metrics import ECE
from sklearn.metrics import brier_score_loss, roc_auc_score

load_dotenv(BACKEND_DIR / ".env")
os.environ["CHROMA_COLLECTION_NAME"] = os.getenv("EVAL_CHROMA_COLLECTION", "squad_eval")

from app.api.router import query_with_uncertainty
from app.core.database import collection
from app.models.schemas import UncertaintyQueryRequest

# ── Constants ────────────────────────────────────────────────────────────────
N_PER_CLASS    = int(os.getenv("EVAL_N_PER_CLASS", "50"))        # 50 answerable + 50 unanswerable = 100 total
RANDOM_SEED    = 42
INTER_QUERY_SLEEP = 6.0   # seconds between queries (~10 req/min)
RETRY_DELAYS   = [5, 10, 20]   # exponential backoff for rate-limit only
MIN_RESULTS    = 10        # abort if fewer than this many succeed
ABSTENTION_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
ABSTENTION_PHRASES = [
    "don't have", "cannot", "insufficient",
    "not enough", "unable to", "no information",
    "not mentioned", "not found",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_answer_and_confidence(res: Any) -> Tuple[str, float]:
    """Extract answer text and predictive_confidence from pipeline response."""
    answer, conf = "", 0.0
    if isinstance(res, tuple):
        answer = str(res[0])
        conf = (
            getattr(res[1], "predictive_confidence", res[1])
            if len(res) > 1 else 0.0
        )
    elif isinstance(res, dict):
        answer = str(res.get("answer", ""))
        c = res.get("confidence")
        conf = (
            c.get("predictive_confidence", 0.0)
            if isinstance(c, dict)
            else (c or res.get("predictive_confidence", 0.0))
        )
    else:
        answer = str(getattr(res, "answer", ""))
        conf_obj = getattr(res, "confidence", None)
        conf = (
            getattr(conf_obj, "predictive_confidence", 0.0)
            if conf_obj is not None
            else getattr(res, "predictive_confidence", 0.0)
        )
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    return answer, max(0.0, min(1.0, conf))


def is_prediction_correct(
    answer: str, gold_answer: Any, is_answerable: bool
) -> int:
    pred_lower = answer.lower()
    if is_answerable:
        if isinstance(gold_answer, list):
            return int(any(str(g).lower() in pred_lower for g in gold_answer if str(g)))
        return int(str(gold_answer).lower() in pred_lower if str(gold_answer) else False)
    return int(any(phrase in pred_lower for phrase in ABSTENTION_PHRASES))


def is_rate_limit_error(e: Exception) -> bool:
    cls = type(e).__name__
    msg = str(e).lower()
    return "ratelimit" in cls.lower() or "rate limit" in msg or "429" in msg


def get_default_doc_id() -> str:
    try:
        res = collection.get(limit=1)
        if res and res.get("metadatas") and len(res["metadatas"]) > 0:
            return res["metadatas"][0].get("doc_id", "default_doc")
    except Exception:
        pass
    return "default_doc"


def load_squad2_balanced() -> List[Dict]:
    """Load 75 answerable + 75 unanswerable from SQuAD 2.0 validation set."""
    import random
    try:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad_v2", split="validation")

        answerable   = []
        unanswerable = []
        for ex in ds:
            texts = ex["answers"]["text"]
            is_ans = len(texts) > 0
            item = {
                "query":         ex["question"],
                "gold_answer":   texts[0] if is_ans else "",
                "is_answerable": is_ans,
                "context":       ex["context"],
            }
            (answerable if is_ans else unanswerable).append(item)

        random.seed(RANDOM_SEED)
        n = min(N_PER_CLASS, len(answerable), len(unanswerable))
        sampled = random.sample(answerable, n) + random.sample(unanswerable, n)
        random.shuffle(sampled)
        print(f"Loaded {len(sampled)} examples from SQuAD 2.0 "
              f"({n} answerable + {n} unanswerable, seed={RANDOM_SEED})")
        return sampled

    except Exception as e:
        print(f"[ERROR] Could not load SQuAD 2.0: {e}")
        print("[ERROR] No synthetic fallback. Fix the dataset issue and re-run.")
        sys.exit(1)


# ── Core eval loop ────────────────────────────────────────────────────────────

async def call_with_retry(req: UncertaintyQueryRequest, idx: int, total: int):
    """Call query_with_uncertainty with rate-limit retry; raises on non-RL errors."""
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            print(f"  [retry {attempt}/{len(RETRY_DELAYS)}] rate-limit hit, "
                  f"sleeping {delay}s ...")
            await asyncio.sleep(delay)
        try:
            return await query_with_uncertainty(req)
        except Exception as e:
            if is_rate_limit_error(e):
                if attempt < len(RETRY_DELAYS):
                    continue          # try again after backoff
                raise               # exhausted retries
            raise                   # non-retryable — propagate immediately


async def evaluate_pipeline():
    ds          = load_squad2_balanced()
    default_doc = get_default_doc_id()
    results     = []
    n_failed    = 0
    total       = len(ds)

    print(f"\nStarting DocuMind evaluation on SQuAD 2.0 "
          f"({total} examples, {INTER_QUERY_SLEEP}s sleep between queries)...\n")

    for idx, item in enumerate(ds):
        query       = item.get("query", "")
        gold_answer = item.get("gold_answer", "")
        is_ans      = bool(item.get("is_answerable", True))
        doc_id      = item.get("doc_id") or default_doc

        req = UncertaintyQueryRequest(doc_id=doc_id, question=query, top_k=5)

        try:
            res = await call_with_retry(req, idx + 1, total)
            pred_answer, pred_conf = extract_answer_and_confidence(res)
            correct = is_prediction_correct(pred_answer, gold_answer, is_ans)

            # Extract gate tracking fields
            decision = getattr(res, "decision", None)
            abstained = getattr(res, "abstained", False)
            cm = getattr(res, "corrective_metadata", None)
            rr_scores = getattr(cm, "rerank_scores", []) if cm else []
            max_rr = max(rr_scores) if rr_scores else None

            # Raw generated answer (pre-gate): if abstained via fallback string, check if raw exists
            deployed_answer = pred_answer
            raw_answer = deployed_answer

            results.append({
                "id":                    idx,
                "query":                 query,
                "gold_answer":           gold_answer,
                "is_answerable":         is_ans,
                "predicted_answer":      deployed_answer,
                "raw_answer":            raw_answer,
                "predictive_confidence": round(pred_conf, 4),
                "decision":              decision,
                "max_rerank_score":      round(max_rr, 4) if max_rr is not None else None,
                "abstained":             abstained,
                "is_correct":            correct,
            })
            status = "CORRECT" if correct else "INCORRECT"
            print(f"[{idx+1:03d}/{total}] {status:<9} | "
                  f"Conf: {pred_conf:.4f} | Dec: {decision} | Q: {query[:40]}")

        except Exception as e:
            n_failed += 1
            print(f"[{idx+1:03d}/{total}] FAILED     | "
                  f"{type(e).__name__}: {str(e)[:100]}")

        # Inter-query pacing (always, even after failure)
        time.sleep(INTER_QUERY_SLEEP)

    print(f"\n{'='*55}")
    print(f"  Run complete: {len(results)} succeeded, {n_failed} failed")
    print(f"{'='*55}")

    if len(results) < MIN_RESULTS:
        print(f"\n[ABORT] Only {len(results)} results — need at least {MIN_RESULTS}. "
              "Fix rate limits and re-run.")
        sys.exit(1)

    # ── Metrics ──────────────────────────────────────────────────────────────
    confidences = np.array([r["predictive_confidence"] for r in results])
    labels      = np.array([r["is_correct"]            for r in results])

    if len(np.unique(confidences)) <= 1:
        print("\n[WARNING] All confidences are identical — EDL may be broken.")

    ece   = float(ECE(bins=10).measure(confidences, labels))
    brier = float(brier_score_loss(labels, confidences))
    acc   = float(np.mean(labels))

    # AUROC both directions
    auroc_pos = auroc_neg = None
    if len(np.unique(labels)) >= 2:
        try:
            auroc_pos = float(roc_auc_score(labels,  confidences))
            auroc_neg = float(roc_auc_score(labels, -confidences))
        except Exception as e:
            print(f"[WARNING] AUROC failed: {e}")

    # Confidence distribution
    conf_std    = float(np.std(confidences))
    n_unique    = int(len(np.unique(confidences)))
    overconf    = int(np.sum(confidences > 0.9))
    underconf   = int(np.sum(confidences < 0.3))

    # ── Print metrics ─────────────────────────────────────────────────────────
    def fmt(v): return f"{v:.4f}" if v is not None else "N/A"

    print(f"\n{'='*55}")
    print(f"  DocuMind SQuAD 2.0 Evaluation Summary (N={len(results)})")
    print(f"{'='*55}")
    print(f"  Accuracy:           {fmt(acc)}")
    print(f"  ECE (10 bins):      {fmt(ece)}")
    print(f"  Brier Score:        {fmt(brier)}")
    print(f"  AUROC (+conf):      {fmt(auroc_pos)}")
    print(f"  AUROC (-conf):      {fmt(auroc_neg)}")
    print(f"  Conf min/max/std:   {fmt(float(np.min(confidences)))} / "
          f"{fmt(float(np.max(confidences)))} / {fmt(conf_std)}")
    print(f"  Unique conf values: {n_unique}")
    print(f"  Overconfident >0.9: {overconf}")
    print(f"  Underconfident<0.3: {underconf}")
    print(f"{'='*55}")

    # ── Abstention analysis ───────────────────────────────────────────────────
    print(f"\n  Abstention analysis (include only examples where conf >= threshold):")
    print(f"  {'Threshold':>10}  {'Coverage':>10}  {'Accuracy':>10}")
    print(f"  {'-'*34}")
    for thr in ABSTENTION_THRESHOLDS:
        mask = confidences >= thr
        n_covered = int(np.sum(mask))
        if n_covered > 0:
            acc_thr = float(np.mean(labels[mask]))
            cov = n_covered / len(results)
            print(f"  {thr:>10.1f}  {cov:>10.1%}  {acc_thr:>10.4f}")
        else:
            print(f"  {thr:>10.1f}  {'0':>10}  {'N/A':>10}")
    print()

    # ── Save results ──────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = BACKEND_DIR / f"eval_results_squad2_fixed_{ts}.json"
    summary  = {
        "timestamp": ts,
        "n_succeeded": len(results),
        "n_failed":    n_failed,
        "metrics": {
            "accuracy":    round(acc,   4),
            "ece":         round(ece,   4),
            "brier_score": round(brier, 4),
            "auroc_pos":   round(auroc_pos, 4) if auroc_pos is not None else None,
            "auroc_neg":   round(auroc_neg, 4) if auroc_neg is not None else None,
            "conf_std":    round(conf_std, 4),
            "n_unique_conf": n_unique,
        },
        "results": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Raw results saved to {out_file}\n")

    # Also overwrite the legacy file (for diagnose_eval.py compat)
    legacy = BACKEND_DIR / "eval_rgb_results.json"
    with open(legacy, "w", encoding="utf-8") as f:
        json.dump({"metrics": summary["metrics"], "results": results},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())