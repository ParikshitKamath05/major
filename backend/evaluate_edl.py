"""
evaluate_edl.py
Evaluation & comparison script: Linear Weighted Fusion vs. EDL Fusion.

Computes calibration metrics:
    - ECE   (Expected Calibration Error)
    - Brier Score
    - NLL   (Negative Log-Likelihood)
    - AUROC (Area Under ROC for error detection)
    - AUPRC (Area Under Precision–Recall Curve)
    - Selective Risk / Coverage curves

Usage:
    python evaluate_edl.py --data evaluation_data.jsonl [--output results.json]

Input format (JSONL):
    {"query": "...", "signals": {"semantic_confidence": 0.8,
     "self_consistency": 0.7, "retrieval_score": 0.6}, "is_correct": 1}

Or use uncertainty_logs.jsonl with ground-truth labels added.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import the fusion modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidential import EvidentialFusion


# ---------------------------------------------------------------------------
# LEGACY LINEAR FUSION (for comparison)
# ---------------------------------------------------------------------------

def legacy_fuse_scores(
    verbalized_confidence: float,
    self_consistency: float,
    retrieval_score: float,
) -> float:
    """
    Original linear weighted fusion:
        C_fused = 0.40 × C_verbalized + 0.35 × C_self-consistency + 0.25 × C_retrieval
    """
    score = (
        0.40 * verbalized_confidence
        + 0.35 * self_consistency
        + 0.25 * retrieval_score
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# CALIBRATION METRICS
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ (|B_m| / N) × |accuracy(B_m) − confidence(B_m)|

    Lower is better. Perfect calibration → ECE = 0.
    """
    if not confidences:
        return 0.0

    N = len(confidences)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for m in range(n_bins):
        lo, hi = bin_boundaries[m], bin_boundaries[m + 1]
        in_bin = [
            (c, l) for c, l in zip(confidences, labels)
            if lo <= c < (hi if m < n_bins - 1 else hi + 1e-6)
        ]
        if not in_bin:
            continue

        bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
        bin_acc = sum(l for _, l in in_bin) / len(in_bin)
        ece += (len(in_bin) / N) * abs(bin_acc - bin_conf)

    return ece


def brier_score(confidences: List[float], labels: List[int]) -> float:
    """
    Brier Score = (1/N) Σ (confidence_i − label_i)².

    Lower is better. Range [0, 1].
    """
    if not confidences:
        return 0.0
    return sum(
        (c - l) ** 2 for c, l in zip(confidences, labels)
    ) / len(confidences)


def negative_log_likelihood(
    confidences: List[float],
    labels: List[int],
    eps: float = 1e-12,
) -> float:
    """
    Negative Log-Likelihood = -(1/N) Σ [y·log(p) + (1-y)·log(1-p)].

    Lower is better.
    """
    if not confidences:
        return 0.0
    nll = 0.0
    for c, l in zip(confidences, labels):
        c = max(eps, min(1.0 - eps, c))
        nll -= l * math.log(c) + (1 - l) * math.log(1.0 - c)
    return nll / len(confidences)


def auroc(
    scores: List[float],
    labels: List[int],
) -> float:
    """
    Area Under the ROC Curve for error detection.

    'scores' should be higher for items we predict are CORRECT.
    'labels' are 1=correct, 0=incorrect.

    Computed via the Wilcoxon-Mann-Whitney statistic (O(n·m)).
    """
    positives = [s for s, l in zip(scores, labels) if l == 1]
    negatives = [s for s, l in zip(scores, labels) if l == 0]

    if not positives or not negatives:
        return 0.5  # uninformative

    count = 0
    for p in positives:
        for n in negatives:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5

    return count / (len(positives) * len(negatives))


def auprc(
    scores: List[float],
    labels: List[int],
) -> float:
    """
    Area Under the Precision-Recall Curve for error detection.

    Approximate via thresholded precision-recall at 100 thresholds.
    """
    if not scores:
        return 0.0

    thresholds = sorted(set(scores), reverse=True)
    precisions = []
    recalls = []
    total_positive = sum(labels)

    if total_positive == 0:
        return 0.0

    for tau in thresholds:
        predicted_positive = [
            (s, l) for s, l in zip(scores, labels) if s >= tau
        ]
        if not predicted_positive:
            continue

        tp = sum(l for _, l in predicted_positive)
        precision = tp / len(predicted_positive)
        recall = tp / total_positive

        precisions.append(precision)
        recalls.append(recall)

    if len(recalls) < 2:
        return precisions[0] if precisions else 0.0

    # Trapezoidal integration
    area = 0.0
    for i in range(1, len(recalls)):
        area += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0

    return abs(area)


# ---------------------------------------------------------------------------
# SELECTIVE RISK / COVERAGE
# ---------------------------------------------------------------------------

def selective_risk_coverage(
    confidences: List[float],
    labels: List[int],
    n_points: int = 50,
) -> List[Dict[str, float]]:
    """
    Compute selective risk at different coverage levels.

    Coverage = fraction of queries we answer (confidence ≥ threshold).
    Selective Risk = error rate among answered queries.

    Returns list of {coverage, risk, threshold} dicts.
    """
    if not confidences:
        return []

    sorted_pairs = sorted(
        zip(confidences, labels), key=lambda x: x[0], reverse=True
    )

    results = []
    total = len(sorted_pairs)

    step = max(1, total // n_points)
    for k in range(step, total + 1, step):
        answered = sorted_pairs[:k]
        coverage = k / total
        n_wrong = sum(1 for _, l in answered if l == 0)
        risk = n_wrong / k

        threshold = answered[-1][0] if answered else 0.0

        results.append({
            "coverage": round(coverage, 4),
            "risk": round(risk, 4),
            "threshold": round(threshold, 4),
        })

    return results


# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    method: str
    n_samples: int
    ece: float
    brier: float
    nll: float
    auroc_score: float
    auprc_score: float
    mean_confidence: float
    accuracy: float
    selective_risk_coverage: List[Dict[str, float]]

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_method(
    name: str,
    confidences: List[float],
    labels: List[int],
) -> EvaluationResult:
    """Run all calibration metrics for a single method."""
    return EvaluationResult(
        method=name,
        n_samples=len(confidences),
        ece=round(expected_calibration_error(confidences, labels), 6),
        brier=round(brier_score(confidences, labels), 6),
        nll=round(negative_log_likelihood(confidences, labels), 6),
        auroc_score=round(auroc(confidences, labels), 6),
        auprc_score=round(auprc(confidences, labels), 6),
        mean_confidence=round(
            sum(confidences) / len(confidences) if confidences else 0.0, 4
        ),
        accuracy=round(
            sum(labels) / len(labels) if labels else 0.0, 4
        ),
        selective_risk_coverage=selective_risk_coverage(confidences, labels),
    )


def load_evaluation_data(
    path: str,
) -> List[Tuple[Dict[str, float], int]]:
    """
    Load evaluation data from JSONL.

    Expected format:
        {"signals": {"semantic_confidence": 0.8, "self_consistency": 0.7,
         "retrieval_score": 0.6}, "is_correct": 1}

    Or from uncertainty logs with added labels:
        {"signals": {..., "semantic_confidence": ..., "self_consistency": ...,
         "retrieval_score": ...}, "is_correct": 1}
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)

                # Extract signals
                signals = record.get("signals", record)

                # Try to find the three signals
                semantic_conf = float(
                    signals.get("semantic_confidence",
                    signals.get("verbalized_confidence", 0.5))
                )
                self_consistency = float(
                    signals.get("self_consistency", 0.5)
                )
                retrieval_score = float(
                    signals.get("retrieval_score", 0.5)
                )

                label = int(record.get("is_correct", record.get("label", 0)))

                data.append((
                    {
                        "semantic_confidence": semantic_conf,
                        "self_consistency": self_consistency,
                        "retrieval_score": retrieval_score,
                    },
                    label,
                ))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"  [warn] skipping line {line_num}: {e}")
                continue

    return data


def run_comparison(data: List[Tuple[Dict[str, float], int]]) -> Dict:
    """
    Run both fusion methods on the same data and compare.

    Returns a dictionary with results for both methods.
    """
    edl_fusion = EvidentialFusion(evidence_scale=20.0)

    legacy_confidences = []
    edl_confidences = []
    labels = []

    for signals, label in data:
        sc = signals["semantic_confidence"]
        si = signals["self_consistency"]
        rt = signals["retrieval_score"]

        # Legacy: linear weighted fusion
        legacy_conf = legacy_fuse_scores(sc, si, rt)
        legacy_confidences.append(legacy_conf)

        # EDL: Dirichlet evidence fusion
        edl_result = edl_fusion.fuse(sc, si, rt)
        edl_confidences.append(edl_result.predictive_confidence)

        labels.append(label)

    # Evaluate both
    legacy_eval = evaluate_method("Linear Weighted Fusion", legacy_confidences, labels)
    edl_eval = evaluate_method("Evidential Deep Learning (EDL)", edl_confidences, labels)

    # Compute deltas
    deltas = {}
    for metric in ["ece", "brier", "nll", "auroc_score", "auprc_score"]:
        legacy_val = getattr(legacy_eval, metric)
        edl_val = getattr(edl_eval, metric)
        improvement = legacy_val - edl_val
        deltas[metric] = {
            "legacy": legacy_val,
            "edl": edl_val,
            "delta": round(improvement, 6),
            "improved": (
                improvement > 0
                if metric in ("ece", "brier", "nll")
                else improvement < 0  # AUROC/AUPRC: higher is better
            ),
        }

    return {
        "n_samples": len(data),
        "legacy": legacy_eval.as_dict(),
        "edl": edl_eval.as_dict(),
        "comparison": deltas,
    }


def print_comparison_table(results: Dict) -> None:
    """Pretty-print the comparison results."""
    print("\n" + "=" * 72)
    print("  EVALUATION: Linear Weighted Fusion vs. Evidential Deep Learning")
    print("=" * 72)
    print(f"  Samples: {results['n_samples']}")
    print()

    header = f"  {'Metric':<30} {'Legacy':>12} {'EDL':>12} {'Delta':>12} {'Better?':>8}"
    print(header)
    print("  " + "-" * 74)

    metrics = {
        "ECE ↓": "ece",
        "Brier Score ↓": "brier",
        "NLL ↓": "nll",
        "AUROC ↑": "auroc_score",
        "AUPRC ↑": "auprc_score",
    }

    for display_name, key in metrics.items():
        comp = results["comparison"][key]
        better = "✓ EDL" if comp["improved"] else "✗ Legacy"
        print(
            f"  {display_name:<30} {comp['legacy']:>12.6f} "
            f"{comp['edl']:>12.6f} {comp['delta']:>+12.6f} {better:>8}"
        )

    print()
    print(f"  Legacy Mean Confidence: {results['legacy']['mean_confidence']:.4f}")
    print(f"  EDL Mean Confidence:    {results['edl']['mean_confidence']:.4f}")
    print(f"  Accuracy:               {results['legacy']['accuracy']:.4f}")
    print("=" * 72)


def generate_sample_data(output_path: str, n: int = 200) -> None:
    """Generate synthetic evaluation data for testing the script."""
    import random
    random.seed(42)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            # Generate correlated signals
            true_quality = random.random()

            semantic_conf = max(0.0, min(1.0,
                true_quality + random.gauss(0, 0.15)
            ))
            self_consistency = max(0.0, min(1.0,
                true_quality + random.gauss(0, 0.12)
            ))
            retrieval_score = max(0.0, min(1.0,
                true_quality + random.gauss(0, 0.18)
            ))

            # Label: correct if quality is decent (with noise)
            is_correct = int(true_quality + random.gauss(0, 0.1) > 0.4)

            record = {
                "signals": {
                    "semantic_confidence": round(semantic_conf, 4),
                    "self_consistency": round(self_consistency, 4),
                    "retrieval_score": round(retrieval_score, 4),
                },
                "is_correct": is_correct,
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Generated {n} synthetic samples → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Linear Fusion vs. Evidential Deep Learning"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to evaluation data JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--generate-sample",
        type=str,
        default=None,
        metavar="PATH",
        help="Generate synthetic evaluation data and save to PATH",
    )

    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_data(args.generate_sample)
        if not args.data:
            args.data = args.generate_sample

    if not args.data:
        print("Error: --data is required (or use --generate-sample)")
        sys.exit(1)

    print(f"\n  Loading data from: {args.data}")
    data = load_evaluation_data(args.data)
    print(f"  Loaded {len(data)} evaluation samples")

    if len(data) < 10:
        print("  Warning: very few samples. Results may not be meaningful.")

    results = run_comparison(data)
    print_comparison_table(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
