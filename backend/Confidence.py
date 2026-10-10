"""
Confidence.py
Advanced Uncertainty Estimation Pipeline for DocuMind RAG.

Architecture (v2 — Semantic Entropy + Evidential Fusion):

    ┌──────────────────────────────────────────────────────────┐
    │  Query + Context                                         │
    │       │                                                  │
    │       ├─► Generate N candidate answers (Groq)            │
    │       │       │                                          │
    │       │       ├─► Embed all candidates (Cohere)          │
    │       │       │       │                                  │
    │       │       │       ├─► Cluster semantically ──► SE    │
    │       │       │       │       (epistemic uncertainty)    │
    │       │       │       │       (aleatoric uncertainty)    │
    │       │       │       │                                  │
    │       │       │       └─► Pairwise similarity ──► SC     │
    │       │       │           (self-consistency score)       │
    │       │       │                                          │
    │       ├─► Retrieval distances ──► RS                     │
    │       │       (retrieval quality score)                  │
    │       │                                                  │
    │       └─► EDL Fusion(SE, SC, RS)                         │
    │               ├─► Dirichlet parameters                   │
    │               ├─► Belief masses                          │
    │               ├─► Predictive confidence                  │
    │               └─► Epistemic uncertainty (vacuity)        │
    └──────────────────────────────────────────────────────────┘

Changes from v1:
    - Verbalized confidence REMOVED (saturated, unreliable)
    - Semantic entropy ADDED (clusters sampled answers by meaning)
    - Linear fusion REPLACED with Dirichlet-based EDL fusion
    - Candidates shared between SE and SC (single embedding call)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from itertools import combinations
from typing import List, Optional, Sequence, Dict, Any, Tuple

from dotenv import load_dotenv
from groq import AsyncGroq, APIStatusError, RateLimitError

from evidential import EvidentialFusion, EvidentialResult

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SEMANTIC_ENTROPY_MODEL = os.getenv(
    "GROQ_SELF_CONSISTENCY_MODEL", "qwen/qwen3.6-27b"
).strip()

# Number of candidate answers to sample (shared by SE and SC)
SEMANTIC_SAMPLES_N = int(os.getenv("SEMANTIC_SAMPLES_N", "5"))

# Sampling temperature for candidate generation
SEMANTIC_TEMPERATURE = float(os.getenv("SEMANTIC_TEMPERATURE", "0.7"))

# Cosine-similarity threshold for semantic clustering
SEMANTIC_CLUSTER_THRESHOLD = float(
    os.getenv("SEMANTIC_CLUSTER_THRESHOLD", "0.75")
)

# Retrieval relevance threshold
RETRIEVAL_RELEVANCE_THRESHOLD = float(
    os.getenv("RETRIEVAL_RELEVANCE_THRESHOLD", "0.40")
)

# Fallback values when computation is impossible
FALLBACK_SELF_CONSISTENCY = 0.5
FALLBACK_SEMANTIC_CONFIDENCE = 0.5

# EDL evidence scale (global multiplier for Dirichlet strength)
EDL_EVIDENCE_SCALE = float(os.getenv("EDL_EVIDENCE_SCALE", "20.0"))

# Instantiate the EDL fusion engine
_edl_fusion = EvidentialFusion(evidence_scale=EDL_EVIDENCE_SCALE)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# RETRIEVAL SIGNAL
# ---------------------------------------------------------------------------

@dataclass
class RetrievalStats:
    mean_similarity: float
    top1_similarity: float
    top1_top2_margin: float
    std_similarity: float
    fraction_relevant: float
    retrieval_score: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_retrieval_stats(
    distances: Sequence[float],
    relevance_threshold: float = RETRIEVAL_RELEVANCE_THRESHOLD,
) -> RetrievalStats:
    if not distances:
        return RetrievalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    similarities = [_clamp01(1.0 - distance) for distance in distances]
    mean_similarity = sum(similarities) / len(similarities)

    sorted_similarities = sorted(similarities, reverse=True)
    top1_similarity = sorted_similarities[0]
    top1_top2_margin = (
        sorted_similarities[0] - sorted_similarities[1]
        if len(sorted_similarities) > 1
        else sorted_similarities[0]
    )

    variance = sum(
        (s - mean_similarity) ** 2 for s in similarities
    ) / len(similarities)
    std_similarity = math.sqrt(variance)

    fraction_relevant = sum(
        1 for s in similarities if s >= relevance_threshold
    ) / len(similarities)
    stability = _clamp01(1.0 - std_similarity)

    retrieval_score = _clamp01(
        0.50 * top1_similarity
        + 0.30 * mean_similarity
        + 0.10 * fraction_relevant
        + 0.10 * stability
    )

    return RetrievalStats(
        mean_similarity=round(mean_similarity, 4),
        top1_similarity=round(top1_similarity, 4),
        top1_top2_margin=round(top1_top2_margin, 4),
        std_similarity=round(std_similarity, 4),
        fraction_relevant=round(fraction_relevant, 4),
        retrieval_score=round(retrieval_score, 4),
    )


# ---------------------------------------------------------------------------
# TEXT UTILITIES (shared by SE and SC)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
    "were", "is", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "should", "can", "could", "may",
    "might", "must", "of", "to", "in", "on", "at", "by", "it", "as",
    "an", "a", "or", "if", "but", "not", "no", "so", "than", "then",
})


def _tokenize(text: str) -> List[str]:
    return [
        t for t in _WORD_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 2
    ]


def _tf_cosine_similarity(text_a: str, text_b: str) -> float:
    """TF-based cosine similarity (fallback when embeddings unavailable)."""
    tokens_a = Counter(_tokenize(text_a))
    tokens_b = Counter(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0

    shared_vocab = set(tokens_a) | set(tokens_b)
    dot = sum(tokens_a.get(w, 0) * tokens_b.get(w, 0) for w in shared_vocab)
    mag_a = math.sqrt(sum(v * v for v in tokens_a.values()))
    mag_b = math.sqrt(sum(v * v for v in tokens_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0

    return _clamp01(dot / (mag_a * mag_b))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _clamp01(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# CANDIDATE ANSWER GENERATION
# ---------------------------------------------------------------------------

_CANDIDATE_SYSTEM_PROMPT = (
    "You are a precise document analyst. Answer the user's question using ONLY "
    "the provided context. Be concise. If the context does not contain the "
    "answer, say that explicitly."
)

_CANDIDATE_USER_TEMPLATE = """DOCUMENT CHUNKS:
{context}

USER QUESTION:
{query}

Answer using only the chunks above."""


async def get_candidate_answer(
    client: AsyncGroq,
    query: str,
    context: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 1,
) -> Optional[str]:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=SEMANTIC_ENTROPY_MODEL,
                    messages=[
                        {"role": "system", "content": _CANDIDATE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _CANDIDATE_USER_TEMPLATE.format(
                                context=context, query=query
                            ),
                        },
                    ],
                    temperature=SEMANTIC_TEMPERATURE,
                    max_tokens=150,
                )
            return response.choices[0].message.content
        except RateLimitError as e:
            last_error = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(0.75 * (attempt + 1))
        except APIStatusError as e:
            last_error = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(0.5 * (attempt + 1))
        except Exception as e:
            last_error = e
            print(
                f"[confidence] unexpected candidate error: "
                f"{type(e).__name__} - {e}"
            )
            break

    print(
        f"[confidence] candidate answer generation failed: "
        f"{type(last_error).__name__ if last_error else 'Unknown'}"
    )
    return None


# ---------------------------------------------------------------------------
# SEMANTIC ENTROPY (replaces Verbalized Confidence)
# ---------------------------------------------------------------------------

@dataclass
class SemanticEntropyResult:
    """Result from semantic entropy computation."""
    entropy: float                  # Raw Shannon entropy
    normalized_entropy: float       # Normalised to [0, 1]
    epistemic_uncertainty: float    # = normalised_entropy
    aleatoric_uncertainty: float    # Within-cluster divergence
    num_clusters: int               # Number of meaning classes
    cluster_sizes: List[int]        # Sizes of each cluster
    semantic_confidence: float      # 1 − normalised_entropy

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cluster_by_similarity(
    n: int,
    sim_fn,
    threshold: float = SEMANTIC_CLUSTER_THRESHOLD,
) -> Tuple[List[int], List[List[int]]]:
    """
    Greedy agglomerative clustering based on average similarity.

    Args:
        n:          Number of items.
        sim_fn:     Callable(i, j) → float similarity.
        threshold:  Minimum average similarity to join a cluster.

    Returns:
        (assignments, clusters) where assignments[i] = cluster_id
        and clusters[c] = [list of item indices].
    """
    if n == 0:
        return [], []

    clusters: List[List[int]] = [[0]]
    assignments = [0] * n

    for i in range(1, n):
        best_cluster = -1
        best_avg_sim = -1.0

        for c_idx, cluster in enumerate(clusters):
            avg_sim = sum(sim_fn(i, j) for j in cluster) / len(cluster)
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_cluster = c_idx

        if best_avg_sim >= threshold:
            clusters[best_cluster].append(i)
            assignments[i] = best_cluster
        else:
            assignments[i] = len(clusters)
            clusters.append([i])

    return assignments, clusters


def _compute_within_cluster_variance(
    clusters: List[List[int]],
    sim_fn,
) -> float:
    """
    Compute average within-cluster divergence (1 − similarity).

    Higher values indicate more variation within meaning classes
    (aleatoric uncertainty).
    """
    total_divergence = 0.0
    total_weight = 0.0

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        pairs = list(combinations(cluster, 2))
        avg_divergence = sum(
            1.0 - sim_fn(i, j) for i, j in pairs
        ) / len(pairs)

        weight = len(cluster)
        total_divergence += avg_divergence * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return _clamp01(total_divergence / total_weight)


def compute_semantic_entropy(
    answers: List[str],
    embeddings: Optional[List[List[float]]] = None,
) -> SemanticEntropyResult:
    """
    Compute semantic entropy by clustering sampled answers into meaning
    classes and computing Shannon entropy over the cluster distribution.

    Two-channel output:
        - Epistemic uncertainty: entropy over meaning clusters.
          High → model generates semantically diverse answers → doesn't know.
        - Aleatoric uncertainty: within-cluster divergence.
          High → even semantically similar answers differ → inherent ambiguity.

    Args:
        answers:     List of valid candidate answer strings.
        embeddings:  Pre-computed embeddings (same order as answers).
                     Falls back to TF-IDF if None.

    Returns:
        SemanticEntropyResult with both uncertainty channels.
    """
    n = len(answers)

    if n < 2:
        return SemanticEntropyResult(
            entropy=0.0,
            normalized_entropy=0.5 if n < 2 else 0.0,
            epistemic_uncertainty=0.5 if n < 2 else 0.0,
            aleatoric_uncertainty=0.0,
            num_clusters=n,
            cluster_sizes=[1] * n,
            semantic_confidence=FALLBACK_SEMANTIC_CONFIDENCE,
        )

    # Build similarity function (embedding-based or TF-IDF fallback)
    if embeddings is not None and len(embeddings) == n:
        # Pre-compute pairwise similarities for efficiency
        _sim_cache: Dict[Tuple[int, int], float] = {}

        def sim_fn(i: int, j: int) -> float:
            key = (min(i, j), max(i, j))
            if key not in _sim_cache:
                _sim_cache[key] = _cosine_similarity(
                    embeddings[i], embeddings[j]
                )
            return _sim_cache[key]
    else:
        _sim_cache_tf: Dict[Tuple[int, int], float] = {}

        def sim_fn(i: int, j: int) -> float:
            key = (min(i, j), max(i, j))
            if key not in _sim_cache_tf:
                _sim_cache_tf[key] = _tf_cosine_similarity(
                    answers[i], answers[j]
                )
            return _sim_cache_tf[key]

    # Cluster answers into semantic equivalence classes
    _, clusters = _cluster_by_similarity(n, sim_fn, SEMANTIC_CLUSTER_THRESHOLD)

    # Shannon entropy over cluster distribution
    cluster_sizes = [len(c) for c in clusters]
    entropy = 0.0
    for size in cluster_sizes:
        if size > 0:
            p = size / n
            entropy -= p * math.log(p + 1e-12)

    # Normalise: max entropy = log(n) when each answer is its own cluster
    max_entropy = math.log(n) if n > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    normalized_entropy = _clamp01(normalized_entropy)

    # Aleatoric uncertainty: within-cluster divergence
    aleatoric = _compute_within_cluster_variance(clusters, sim_fn)

    # Semantic confidence: inverse of epistemic uncertainty
    semantic_confidence = _clamp01(1.0 - normalized_entropy)

    return SemanticEntropyResult(
        entropy=round(entropy, 4),
        normalized_entropy=round(normalized_entropy, 4),
        epistemic_uncertainty=round(normalized_entropy, 4),
        aleatoric_uncertainty=round(aleatoric, 4),
        num_clusters=len(clusters),
        cluster_sizes=cluster_sizes,
        semantic_confidence=round(semantic_confidence, 4),
    )


# ---------------------------------------------------------------------------
# SELF-CONSISTENCY (now uses shared embeddings)
# ---------------------------------------------------------------------------

def compute_self_consistency(
    answers: List[str],
    embeddings: Optional[List[List[float]]] = None,
) -> float:
    """
    Compute self-consistency as average pairwise similarity of sampled
    answers.

    Uses pre-computed embeddings when available, otherwise falls back
    to TF-IDF cosine similarity.

    Args:
        answers:     List of valid candidate answer strings.
        embeddings:  Pre-computed embeddings (same order as answers).

    Returns:
        Self-consistency score ∈ [0, 1].
    """
    n = len(answers)
    if n < 2:
        return FALLBACK_SELF_CONSISTENCY

    if embeddings is not None and len(embeddings) == n:
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[j])
            for i, j in combinations(range(n), 2)
        ]
    else:
        similarities = [
            _tf_cosine_similarity(answers[i], answers[j])
            for i, j in combinations(range(n), 2)
        ]

    if not similarities:
        return FALLBACK_SELF_CONSISTENCY

    return _clamp01(sum(similarities) / len(similarities))


# ---------------------------------------------------------------------------
# CONFIDENCE SIGNALS (expanded dataclass)
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSignals:
    """
    Complete uncertainty profile for a single RAG query.

    Contains all three signal channels, the EDL fusion result, and
    backward-compatible fields.
    """

    # ── Retrieval Signal ──
    retrieval_similarity_mean: float
    retrieval_score: float
    top1_similarity: float
    top1_top2_margin: float

    # ── Semantic Entropy (replaces verbalized confidence) ──
    semantic_entropy: float             # Raw Shannon entropy
    normalized_semantic_entropy: float  # [0, 1]
    epistemic_uncertainty: float        # = normalised entropy
    aleatoric_uncertainty: float        # Within-cluster divergence
    num_semantic_clusters: int          # Meaning classes found
    semantic_confidence: float          # 1 − epistemic_uncertainty

    # ── Self-Consistency ──
    self_consistency: float

    # ── Evidential Deep Learning Fusion ──
    belief_correct: float               # b₁ = e₁ / S
    belief_incorrect: float             # b₂ = e₂ / S
    evidential_uncertainty: float       # u  = K  / S
    predictive_confidence: float        # E[p_correct] = α₁ / S
    dirichlet_strength: float           # S  = Σα_k
    total_evidence: float               # Σe_k

    # ── Backward Compatibility ──
    verbalized_confidence: float        # = semantic_confidence
    fused_score: float                  # = predictive_confidence

    # ── Detailed Breakdowns ──
    retrieval_stats: Dict[str, float] = field(default_factory=dict)
    semantic_entropy_details: Dict[str, Any] = field(default_factory=dict)
    edl_details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# CONFIDENCE ESTIMATOR (orchestrator)
# ---------------------------------------------------------------------------

class ConfidenceEstimator:
    """
    Orchestrates the full uncertainty estimation pipeline.

    Pipeline:
        1. Generate N candidate answers concurrently (Groq)
        2. Embed all candidates in a single batch (Cohere)
        3. Compute semantic entropy from embeddings (clustering + entropy)
        4. Compute self-consistency from embeddings (pairwise similarity)
        5. Compute retrieval stats from distances (pure math)
        6. Fuse all three signals via EDL (Dirichlet evidence)
        7. Return comprehensive ConfidenceSignals

    Efficiency gains over v1:
        - Removed the verbalized confidence LLM call
        - Embeddings computed once and shared between SE and SC
        - More candidate answers (5 vs 3) for better entropy estimation
    """

    def __init__(
        self,
        client: AsyncGroq,
        cohere_client=None,
        evidential_fusion: Optional[EvidentialFusion] = None,
    ):
        self.client = client
        self.cohere_client = cohere_client
        self.evidential_fusion = evidential_fusion or _edl_fusion
        self.semaphore = asyncio.Semaphore(4)  # Limit concurrent Groq calls

    async def estimate(
        self,
        query: str,
        context: str,
        distances: List[float],
    ) -> ConfidenceSignals:
        """
        Run the full uncertainty estimation pipeline.

        Args:
            query:     User question.
            context:   Concatenated retrieved document chunks.
            distances: ChromaDB distances for retrieved chunks.

        Returns:
            ConfidenceSignals with all uncertainty channels and fusion result.
        """
        # ── 1. Retrieval Stats ──
        retrieval_stats = compute_retrieval_stats(distances)

        # ── 2. Generate N candidate answers concurrently ──
        candidate_tasks = [
            get_candidate_answer(
                self.client, query, context[:6000], self.semaphore
            )
            for _ in range(SEMANTIC_SAMPLES_N)
        ]
        candidates_raw = await asyncio.gather(
            *candidate_tasks, return_exceptions=True
        )
        candidates: List[Optional[str]] = [
            c if isinstance(c, str) else None for c in candidates_raw
        ]

        # Filter to valid candidates (non-empty, > 20 chars)
        valid_answers = [
            c.strip() for c in candidates
            if c and len(c.strip()) > 20
        ]

        # ── 3. Embed all valid candidates (single Cohere API call) ──
        embeddings: Optional[List[List[float]]] = None

        if self.cohere_client is not None and len(valid_answers) >= 2:
            try:
                response = await asyncio.to_thread(
                    self.cohere_client.embed,
                    texts=valid_answers,
                    model="embed-english-light-v3.0",
                    input_type="classification",
                )
                embeddings = response.embeddings
            except Exception as e:
                print(f"[confidence] embedding failed, using TF-IDF: {e}")

        # ── 4. Compute Semantic Entropy ──
        se_result = compute_semantic_entropy(valid_answers, embeddings)

        # ── 5. Compute Self-Consistency (shared embeddings) ──
        sc_score = compute_self_consistency(valid_answers, embeddings)

        # ── 6. EDL Fusion ──
        edl_result = self.evidential_fusion.fuse(
            semantic_confidence=se_result.semantic_confidence,
            self_consistency=sc_score,
            retrieval_score=retrieval_stats.retrieval_score,
        )

        # ── 7. Assemble ConfidenceSignals ──
        return ConfidenceSignals(
            # Retrieval
            retrieval_similarity_mean=retrieval_stats.mean_similarity,
            retrieval_score=retrieval_stats.retrieval_score,
            top1_similarity=retrieval_stats.top1_similarity,
            top1_top2_margin=retrieval_stats.top1_top2_margin,
            # Semantic Entropy
            semantic_entropy=se_result.entropy,
            normalized_semantic_entropy=se_result.normalized_entropy,
            epistemic_uncertainty=se_result.epistemic_uncertainty,
            aleatoric_uncertainty=se_result.aleatoric_uncertainty,
            num_semantic_clusters=se_result.num_clusters,
            semantic_confidence=se_result.semantic_confidence,
            # Self-Consistency
            self_consistency=round(sc_score, 4),
            # EDL
            belief_correct=edl_result.belief_correct,
            belief_incorrect=edl_result.belief_incorrect,
            evidential_uncertainty=edl_result.uncertainty_mass,
            predictive_confidence=edl_result.predictive_confidence,
            dirichlet_strength=edl_result.dirichlet_strength,
            total_evidence=edl_result.total_evidence,
            # Backward Compat
            verbalized_confidence=se_result.semantic_confidence,
            fused_score=edl_result.predictive_confidence,
            # Detailed breakdowns
            retrieval_stats=retrieval_stats.as_dict(),
            semantic_entropy_details=se_result.as_dict(),
            edl_details=edl_result.as_dict(),
        )