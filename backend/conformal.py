"""
conformal.py
Split Conformal Prediction for statistically calibrated RAG abstention.

Replaces the static threshold mechanism:
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    LOW_CONFIDENCE_THRESHOLD  = 0.45

with an adaptive, data-driven approach that provides finite-sample coverage
guarantees:
    P(correct | not abstaining) >= 1 - α

The conformal predictor maintains a calibration store of past
(confidence, is_correct) observations.  From this store it computes
an adaptive threshold using conformal risk control with Hoeffding
UCB correction.

When calibration data is insufficient (< min_calibration_size), the
predictor gracefully falls back to configurable static thresholds.

Reference:
    Angelopoulos, Bates, et al. (2021). "Learn Then Test: Calibrating
    Predictive Models via Hypothesis Testing."
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class ConformalThresholds:
    """Active thresholds produced by the conformal predictor."""
    abstention_threshold: float     # Below this → abstain
    high_confidence_threshold: float  # Above this → high confidence
    is_calibrated: bool              # True if derived from calibration data
    calibration_size: int            # Number of calibration points used
    coverage_guarantee: float        # 1 − α
    effective_alpha: float           # Significance level used

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConformalDecision:
    """Decision output from the conformal predictor."""
    decision: str           # "high_confidence" | "low_confidence" | "abstain"
    confidence: float       # The input predictive confidence
    thresholds: ConformalThresholds
    prediction_set: List[str]  # {"answer"} or {"answer","abstain"} or {"abstain"}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# CONFORMAL PREDICTOR
# ---------------------------------------------------------------------------

class ConformalPredictor:
    """
    Split conformal predictor for RAG selective prediction / abstention.

    Maintains a calibration store (JSONL) of (confidence, is_correct) pairs.
    Uses conformal risk control to compute an adaptive abstention threshold
    with finite-sample statistical guarantees.

    Decision regions:
        confidence >= high_threshold  → "high_confidence"
        abstention_threshold <= confidence < high_threshold → "low_confidence"
        confidence < abstention_threshold → "abstain"

    When calibration data is insufficient, falls back to static defaults.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        delta: float = 0.05,
        calibration_path: Optional[str] = None,
        min_calibration_size: int = 30,
        max_entries: int = 10000,
        fallback_low: float = 0.45,
        fallback_high: float = 0.75,
    ):
        """
        Args:
            alpha:  Significance level. Guarantee: selective risk ≤ α.
            delta:  Confidence parameter for Hoeffding UCB correction.
            calibration_path:  Path to JSONL calibration store.
            min_calibration_size:  Minimum observations before adaptive
                thresholds activate.
            max_entries:  Maximum calibration entries (FIFO eviction).
            fallback_low:   Static low-confidence threshold (bootstrap).
            fallback_high:  Static high-confidence threshold (bootstrap).
        """
        self.alpha = alpha
        self.delta = delta
        self.calibration_path = calibration_path
        self.min_calibration_size = min_calibration_size
        self.max_entries = max_entries
        self.fallback_low = fallback_low
        self.fallback_high = fallback_high

        self._lock = threading.Lock()
        self._calibration_data: List[Tuple[float, int]] = []

        if calibration_path:
            self._load_calibration()

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------

    def predict(self, confidence: float) -> ConformalDecision:
        """
        Make a conformal prediction decision for the given confidence score.

        Args:
            confidence: Predictive confidence from the EDL fusion ∈ [0, 1].

        Returns:
            ConformalDecision with the abstention/answer decision and
            the active thresholds.
        """
        thresholds = self.get_thresholds()

        if confidence >= thresholds.high_confidence_threshold:
            decision = "high_confidence"
            prediction_set = ["answer"]
        elif confidence >= thresholds.abstention_threshold:
            decision = "low_confidence"
            prediction_set = ["answer", "abstain"]
        else:
            decision = "abstain"
            prediction_set = ["abstain"]

        return ConformalDecision(
            decision=decision,
            confidence=round(confidence, 4),
            thresholds=thresholds,
            prediction_set=prediction_set,
        )

    def get_thresholds(self) -> ConformalThresholds:
        """
        Compute the current adaptive thresholds from calibration data.

        Falls back to static thresholds when calibration is insufficient.
        """
        with self._lock:
            data = list(self._calibration_data)

        if len(data) < self.min_calibration_size:
            return ConformalThresholds(
                abstention_threshold=self.fallback_low,
                high_confidence_threshold=self.fallback_high,
                is_calibrated=False,
                calibration_size=len(data),
                coverage_guarantee=1.0 - self.alpha,
                effective_alpha=self.alpha,
            )

        abstention_threshold = self._compute_abstention_threshold(data)
        high_threshold = self._compute_high_confidence_threshold(data)

        # Ensure high >= low (sanity)
        if high_threshold < abstention_threshold:
            high_threshold = abstention_threshold + 0.10

        return ConformalThresholds(
            abstention_threshold=round(abstention_threshold, 4),
            high_confidence_threshold=round(high_threshold, 4),
            is_calibrated=True,
            calibration_size=len(data),
            coverage_guarantee=round(1.0 - self.alpha, 4),
            effective_alpha=self.alpha,
        )

    def add_calibration_point(
        self,
        confidence: float,
        is_correct: bool,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Record a new calibration observation.

        Called either:
            - Automatically after each query (using proxy correctness), or
            - Via user feedback endpoint with ground-truth labels.
        """
        entry_tuple = (float(confidence), int(is_correct))

        with self._lock:
            self._calibration_data.append(entry_tuple)

            # FIFO eviction to bound memory
            if len(self._calibration_data) > self.max_entries:
                self._calibration_data = self._calibration_data[-self.max_entries:]

        # Persist to disk (append)
        if self.calibration_path:
            self._append_to_store(confidence, is_correct, request_id)

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the calibration store."""
        with self._lock:
            data = list(self._calibration_data)

        if not data:
            return {"size": 0, "is_sufficient": False}

        confidences = [c for c, _ in data]
        labels = [l for _, l in data]

        return {
            "size": len(data),
            "is_sufficient": len(data) >= self.min_calibration_size,
            "accuracy": sum(labels) / len(labels) if labels else 0.0,
            "mean_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        }

    # -----------------------------------------------------------------
    # THRESHOLD COMPUTATION
    # -----------------------------------------------------------------

    def _compute_abstention_threshold(
        self,
        data: List[Tuple[float, int]],
    ) -> float:
        """
        Compute the adaptive abstention threshold via conformal risk control.

        Finds the *lowest* threshold τ such that the selective risk
        (fraction of incorrect answers among those with confidence ≥ τ)
        is bounded by α, with Hoeffding UCB finite-sample correction.

        Iterates thresholds from most conservative (highest) to least
        conservative (lowest), accepting the lowest safe threshold to
        maximise coverage.
        """
        # Candidate thresholds: all unique confidence values
        candidates = sorted(set(c for c, _ in data), reverse=True)

        best_threshold = candidates[0] if candidates else self.fallback_low

        for tau in candidates:
            answered = [(c, l) for c, l in data if c >= tau]
            n = len(answered)
            if n < 5:
                continue

            n_wrong = sum(1 for _, l in answered if l == 0)
            empirical_risk = n_wrong / n

            # Hoeffding upper confidence bound
            ucb = empirical_risk + math.sqrt(
                math.log(2.0 / self.delta) / (2.0 * n)
            )

            if ucb <= self.alpha:
                best_threshold = tau   # safe to lower further
            else:
                break                  # risk exceeded, stop

        return best_threshold

    def _compute_high_confidence_threshold(
        self,
        data: List[Tuple[float, int]],
    ) -> float:
        """
        Compute the high-confidence threshold.

        Finds the confidence level above which the empirical accuracy
        exceeds 95% (or 1 − α/2 if α is very small).

        Uses a Hoeffding LCB to ensure finite-sample validity.
        """
        target_accuracy = max(0.95, 1.0 - self.alpha / 2.0)

        candidates = sorted(set(c for c, _ in data))
        best_threshold = candidates[-1] if candidates else self.fallback_high

        for tau in candidates:
            above = [(c, l) for c, l in data if c >= tau]
            n = len(above)
            if n < 5:
                continue

            n_correct = sum(1 for _, l in above if l == 1)
            accuracy = n_correct / n

            # Lower confidence bound
            lcb = accuracy - math.sqrt(
                math.log(2.0 / self.delta) / (2.0 * n)
            )

            if lcb >= target_accuracy:
                best_threshold = tau
                break

        return best_threshold

    # -----------------------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------------------

    def _load_calibration(self) -> None:
        """Load calibration data from the JSONL store."""
        if not self.calibration_path or not os.path.exists(self.calibration_path):
            return

        entries: List[Tuple[float, int]] = []
        try:
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        conf = float(record.get("confidence", 0.0))
                        correct = int(record.get("is_correct", 0))
                        entries.append((conf, correct))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue

            # Keep only the most recent entries
            if len(entries) > self.max_entries:
                entries = entries[-self.max_entries:]

            with self._lock:
                self._calibration_data = entries

            print(
                f"[conformal] loaded {len(entries)} calibration points "
                f"from {self.calibration_path}"
            )
        except Exception as e:
            print(f"[conformal] failed to load calibration: {e}")

    def _append_to_store(
        self,
        confidence: float,
        is_correct: bool,
        request_id: Optional[str] = None,
    ) -> None:
        """Append a single calibration entry to the JSONL file."""
        try:
            record = {
                "confidence": round(float(confidence), 6),
                "is_correct": int(is_correct),
            }
            if request_id:
                record["request_id"] = request_id

            os.makedirs(os.path.dirname(self.calibration_path), exist_ok=True)
            with open(self.calibration_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[conformal] failed to write calibration entry: {e}")

    # -----------------------------------------------------------------
    # NONCONFORMITY SCORE UTILITIES (for external evaluation)
    # -----------------------------------------------------------------

    @staticmethod
    def nonconformity_score(confidence: float) -> float:
        """Standard nonconformity score: α = 1 − confidence."""
        return 1.0 - max(0.0, min(1.0, confidence))

    def compute_p_value(self, confidence: float) -> float:
        """
        Compute the conformal p-value for a new observation.

        p = (1 + #{s_i >= s_new}) / (n + 1)

        where s_i are calibration nonconformity scores for *correct*
        predictions and s_new is the nonconformity score of the new
        observation.
        """
        with self._lock:
            data = list(self._calibration_data)

        correct_scores = sorted(
            [1.0 - c for c, l in data if l == 1]
        )

        if not correct_scores:
            return 0.5  # uninformative

        s_new = 1.0 - confidence
        n = len(correct_scores)
        count_geq = sum(1 for s in correct_scores if s >= s_new)

        return (1.0 + count_geq) / (n + 1.0)
