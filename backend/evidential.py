"""
evidential.py
Dirichlet-based Evidential Deep Learning (EDL) for calibrated uncertainty fusion.

Replaces the linear weighted fusion formula:
    C_fused = 0.40 × C_verbalized + 0.35 × C_self-consistency + 0.25 × C_retrieval

with a principled Dirichlet evidence-based fusion that models uncertainty as an
explicit distribution over beliefs. Each signal contributes evidence to a
Dirichlet distribution, from which we derive:
    - Belief masses (b_correct, b_incorrect, u)
    - Predictive confidence E[p_correct] = α_correct / S
    - Epistemic uncertainty (vacuity) = K / S
    - Calibrated abstention decisions

Reference:
    Sensoy, Kaplan, Kandemir (2018). "Evidential Deep Learning to Quantify
    Classification Uncertainty." NeurIPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class EvidentialResult:
    """Complete result from evidential fusion."""

    # Evidence totals
    evidence_correct: float
    evidence_incorrect: float
    total_evidence: float

    # Dirichlet parameters: α_k = e_k + 1
    alpha_correct: float
    alpha_incorrect: float
    dirichlet_strength: float   # S = Σ α_k

    # Belief masses (Subjective Logic)
    belief_correct: float       # b₁ = e₁ / S
    belief_incorrect: float     # b₂ = e₂ / S
    uncertainty_mass: float     # u  = K  / S  (vacuity)

    # Predictions
    predictive_confidence: float      # E[p_correct] = α₁ / S
    epistemic_uncertainty_edl: float  # vacuity = K / S

    # Per-signal evidence breakdown
    signal_evidence: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# EVIDENTIAL FUSION ENGINE
# ---------------------------------------------------------------------------

class EvidentialFusion:
    """
    Dirichlet-based evidential fusion for binary classification
    (answerable vs. not-answerable).

    Each uncertainty signal s ∈ [0, 1] is mapped to evidence through a
    non-linear power transform:

        e_correct   = weight × s^γ      × scale
        e_incorrect = weight × (1-s)^γ   × scale

    Evidence is accumulated across signals, then used to parameterise a
    Dirichlet distribution Dir(α₁, α₂) where α_k = e_k + 1.

    Key properties compared to linear fusion:
        1. Non-linear signal interactions via the γ exponent
        2. Explicit epistemic uncertainty through the vacuity mass u = K/S
        3. Proper separation of aleatoric vs. epistemic uncertainty
        4. Principled abstention: abstain when u exceeds a threshold
    """

    # Default signal configuration: name → {weight, gamma}
    DEFAULT_SIGNAL_CONFIG: Dict[str, Dict[str, float]] = {
        "semantic":         {"weight": 0.40, "gamma": 2.0},
        "self_consistency": {"weight": 0.35, "gamma": 2.0},
        "retrieval":        {"weight": 0.25, "gamma": 1.5},
    }

    K = 2  # Binary: answerable / not-answerable

    def __init__(
        self,
        evidence_scale: float = 20.0,
        signal_config: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Args:
            evidence_scale: Global multiplier controlling total Dirichlet
                strength. Higher values → more concentrated posterior →
                lower vacuity for the same signal strengths.
            signal_config: Per-signal {weight, gamma} overrides.
        """
        self.evidence_scale = evidence_scale
        self.signal_config = signal_config or dict(self.DEFAULT_SIGNAL_CONFIG)

    # -----------------------------------------------------------------
    # CORE FUSION
    # -----------------------------------------------------------------

    @staticmethod
    def _signal_to_evidence(
        signal_value: float,
        weight: float,
        gamma: float,
        scale: float,
    ) -> tuple[float, float]:
        """
        Map a [0, 1] signal to evidence for the two classes.

        A signal of 1.0 contributes maximum evidence for *correct*;
        a signal of 0.0 contributes maximum evidence for *incorrect*.
        Intermediate values contribute to both, with γ > 1 dampening
        mid-range signals (making extreme signals more decisive).
        """
        s = max(0.0, min(1.0, float(signal_value)))
        e_correct   = weight * (s ** gamma)       * scale
        e_incorrect = weight * ((1.0 - s) ** gamma) * scale
        return e_correct, e_incorrect

    def fuse(
        self,
        semantic_confidence: float,
        self_consistency: float,
        retrieval_score: float,
    ) -> EvidentialResult:
        """
        Fuse three uncertainty signals via Dirichlet evidence accumulation.

        Args:
            semantic_confidence: 1 − normalised_semantic_entropy ∈ [0, 1].
                High = model generates semantically consistent answers.
            self_consistency: pairwise similarity of sampled answers ∈ [0, 1].
                High = sampled answers agree on surface level.
            retrieval_score: weighted retrieval quality score ∈ [0, 1].
                High = retrieved documents are relevant and diverse.

        Returns:
            EvidentialResult containing the full Dirichlet analysis.
        """
        signals = {
            "semantic":         semantic_confidence,
            "self_consistency": self_consistency,
            "retrieval":        retrieval_score,
        }

        total_e_correct   = 0.0
        total_e_incorrect = 0.0
        signal_evidence: Dict[str, Dict[str, float]] = {}

        for name, value in signals.items():
            cfg = self.signal_config[name]
            e_c, e_i = self._signal_to_evidence(
                value, cfg["weight"], cfg["gamma"], self.evidence_scale,
            )
            total_e_correct   += e_c
            total_e_incorrect += e_i
            signal_evidence[name] = {
                "input_value":        round(value, 4),
                "evidence_correct":   round(e_c, 4),
                "evidence_incorrect": round(e_i, 4),
            }

        # Dirichlet parameters
        alpha_correct   = total_e_correct + 1.0
        alpha_incorrect = total_e_incorrect + 1.0
        S = alpha_correct + alpha_incorrect       # Dirichlet strength

        # Belief masses (Subjective Logic)
        belief_correct   = total_e_correct / S
        belief_incorrect = total_e_incorrect / S
        uncertainty_mass = self.K / S             # vacuity

        # Expected probability of the "correct/answerable" class
        predictive_confidence = alpha_correct / S

        # Epistemic uncertainty = vacuity
        epistemic_edl = self.K / S

        return EvidentialResult(
            evidence_correct       = round(total_e_correct, 4),
            evidence_incorrect     = round(total_e_incorrect, 4),
            total_evidence         = round(total_e_correct + total_e_incorrect, 4),
            alpha_correct          = round(alpha_correct, 4),
            alpha_incorrect        = round(alpha_incorrect, 4),
            dirichlet_strength     = round(S, 4),
            belief_correct         = round(belief_correct, 4),
            belief_incorrect       = round(belief_incorrect, 4),
            uncertainty_mass       = round(uncertainty_mass, 4),
            predictive_confidence  = round(predictive_confidence, 4),
            epistemic_uncertainty_edl = round(epistemic_edl, 4),
            signal_evidence        = signal_evidence,
        )

    # -----------------------------------------------------------------
    # LOSS FUNCTIONS (for future fine-tuning / training)
    # -----------------------------------------------------------------

    @staticmethod
    def edl_mse_loss(
        alphas: List[float],
        target_index: int,
        kl_weight: float = 0.01,
        epoch_fraction: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute the MSE-based EDL loss with KL-divergence regularisation.

        This is the squared-error variant from Sensoy et al. (2018) that
        does NOT require the digamma function:

            L_mse = Σ_k (y_k − p̂_k)² + p̂_k (1 − p̂_k) / (S + 1)

        where p̂_k = α_k / S and y is the one-hot target.

        The KL regularisation penalises divergence from Dir(1, …, 1)
        (uniform Dirichlet) for the *incorrect* classes, preventing the
        model from placing excessive evidence on wrong answers.

        Args:
            alphas:         Dirichlet concentration parameters [α₁, α₂, …].
            target_index:   Index of the true class (0-based).
            kl_weight:      Regularisation strength λ.
            epoch_fraction: Annealing factor (linearly increase kl_weight
                            from 0 → kl_weight over training).

        Returns:
            Dictionary with 'mse_loss', 'kl_loss', and 'total_loss'.
        """
        K = len(alphas)
        S = sum(alphas)

        # One-hot target
        y = [1.0 if k == target_index else 0.0 for k in range(K)]

        # Expected probabilities
        p_hat = [a / S for a in alphas]

        # MSE + variance term
        mse = 0.0
        for k in range(K):
            err_sq = (y[k] - p_hat[k]) ** 2
            var    = p_hat[k] * (1.0 - p_hat[k]) / (S + 1.0)
            mse   += err_sq + var

        # KL regularisation: remove evidence for the correct class
        alpha_tilde = [
            alphas[k] - (alphas[k] - 1.0) * y[k]   # keep only incorrect evidence
            for k in range(K)
        ]
        kl = EvidentialFusion._kl_dirichlet_uniform(alpha_tilde, K)

        # Anneal the KL weight
        effective_kl_weight = kl_weight * epoch_fraction

        total = mse + effective_kl_weight * kl

        return {
            "mse_loss":   round(mse, 6),
            "kl_loss":    round(kl, 6),
            "total_loss": round(total, 6),
        }

    @staticmethod
    def _kl_dirichlet_uniform(alphas: List[float], K: int) -> float:
        """
        KL divergence KL( Dir(α) || Dir(1, …, 1) ).

        Uses the Stirling approximation for the log-gamma ratio when
        scipy is unavailable:
            ln Γ(x) ≈ (x − 0.5) ln x − x + 0.5 ln(2π) + 1/(12x)

        This is accurate to O(1/x²) and works well for x ≥ 1.
        """
        def _log_gamma_approx(x: float) -> float:
            """Stirling approximation of ln Γ(x)."""
            if x <= 0:
                return 0.0
            if x < 1:
                # Use Γ(x) = Γ(x+1) / x
                return _log_gamma_approx(x + 1.0) - math.log(x)
            return (
                (x - 0.5) * math.log(x)
                - x
                + 0.5 * math.log(2.0 * math.pi)
                + 1.0 / (12.0 * x)
            )

        def _digamma_approx(x: float) -> float:
            """Asymptotic approximation of ψ(x) = d/dx ln Γ(x)."""
            if x < 1e-6:
                return -1.0 / max(x, 1e-12)
            # Recurrence: ψ(x) = ψ(x+1) - 1/x  (shift to large x)
            result = 0.0
            while x < 8.0:
                result -= 1.0 / x
                x += 1.0
            # Asymptotic expansion for large x
            result += (
                math.log(x)
                - 1.0 / (2.0 * x)
                - 1.0 / (12.0 * x * x)
                + 1.0 / (120.0 * x ** 4)
            )
            return result

        S_alpha = sum(alphas)
        S_beta  = float(K)  # Dir(1, …, 1) has all params = 1

        kl  = _log_gamma_approx(S_alpha) - _log_gamma_approx(S_beta)
        kl -= sum(_log_gamma_approx(a) for a in alphas)
        kl += sum(_log_gamma_approx(1.0) for _ in range(K))   # = 0

        for k in range(K):
            kl += (alphas[k] - 1.0) * (_digamma_approx(alphas[k]) - _digamma_approx(S_alpha))

        return max(kl, 0.0)  # clamp numerical noise
