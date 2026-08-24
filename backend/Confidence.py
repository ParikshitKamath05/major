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
from typing import List, Optional, Sequence, Dict, Any

from dotenv import load_dotenv
from groq import AsyncGroq, APIStatusError, RateLimitError

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

VERBALIZED_CONFIDENCE_MODEL = os.getenv("GROQ_VERBALIZED_MODEL", "openai/gpt-oss-20b").strip()
SELF_CONSISTENCY_MODEL = os.getenv("GROQ_SELF_CONSISTENCY_MODEL", "qwen/qwen3.6-27b").strip()

SELF_CONSISTENCY_N = 3
SELF_CONSISTENCY_TEMPERATURE = 0.7

FUSION_WEIGHTS = {
    "verbalized": 0.40,
    "self_consistency": 0.35,
    "retrieval": 0.25,
}

FALLBACK_VERBALIZED_CONFIDENCE = 0.5
FALLBACK_SELF_CONSISTENCY = 0.5
RETRIEVAL_RELEVANCE_THRESHOLD = 0.55

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

# -----------------------------------------------------------------------------
# RETRIEVAL SIGNAL
# -----------------------------------------------------------------------------

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
    top1_top2_margin = sorted_similarities[0] - sorted_similarities[1] if len(sorted_similarities) > 1 else sorted_similarities[0]
    
    variance = sum((s - mean_similarity) ** 2 for s in similarities) / len(similarities)
    std_similarity = math.sqrt(variance)
    
    fraction_relevant = sum(1 for s in similarities if s >= relevance_threshold) / len(similarities)
    stability = _clamp01(1.0 - std_similarity)

    retrieval_score = _clamp01(
        0.40 * top1_similarity +
        0.25 * mean_similarity +
        0.15 * _clamp01(top1_top2_margin) +
        0.10 * fraction_relevant +
        0.10 * stability
    )

    return RetrievalStats(
        mean_similarity=round(mean_similarity, 4),
        top1_similarity=round(top1_similarity, 4),
        top1_top2_margin=round(top1_top2_margin, 4),
        std_similarity=round(std_similarity, 4),
        fraction_relevant=round(fraction_relevant, 4),
        retrieval_score=round(retrieval_score, 4),
    )

# -----------------------------------------------------------------------------
# VERBALIZED CONFIDENCE
# -----------------------------------------------------------------------------

_VERBALIZED_PROMPT = """You are an uncertainty estimator for a retrieval-augmented generation system.
You will be given a user query and retrieved document chunks.
Judge whether the retrieved context contains enough information to answer the query accurately.
Rules:
- Do not answer the query.
- Do not explain.
- Return ONLY valid JSON.
- The JSON must have one key: "confidence".
- The value must be a number between 0.0 and 1.0.

QUERY:
{query}

CONTEXT:
{context}

Return ONLY JSON like:
{{"confidence": 0.0}}
"""

_FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

def _parse_confidence_float(raw_text: Optional[str]) -> Optional[float]:
    if not raw_text: return None
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            for key in ("confidence", "score", "value"):
                if key in data: return _clamp01(float(data[key]))
        if isinstance(data, (int, float)): return _clamp01(float(data))
    except Exception: pass

    json_match = _JSON_OBJECT_RE.search(raw_text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                for key in ("confidence", "score", "value"):
                    if key in data: return _clamp01(float(data[key]))
        except Exception: pass

    float_match = _FLOAT_RE.search(raw_text)
    if not float_match: return None
    
    try: value = float(float_match.group(1))
    except ValueError: return None

    if value > 1.0:
        value = value / 10.0 if value <= 10.0 else value / 100.0 if value <= 100.0 else 1.0
        
    return _clamp01(value)

async def get_verbalized_confidence(
    client: AsyncGroq,
    query: str,
    context: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 1,
) -> float:
    prompt = _VERBALIZED_PROMPT.format(query=query, context=context[:6000])
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=VERBALIZED_CONFIDENCE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=30,
                )
            raw = response.choices[0].message.content
            parsed = _parse_confidence_float(raw)
            if parsed is not None:
                return parsed
            last_error = ValueError(f"Could not parse: {raw!r}")
        except RateLimitError as e:
            last_error = e
            if attempt >= max_retries: break
            await asyncio.sleep(0.5 * (2 ** attempt) + random.uniform(0.0, 0.25))
        except APIStatusError as e:
            last_error = e
            if attempt >= max_retries: break
            await asyncio.sleep(0.35 * (2 ** attempt) + random.uniform(0.0, 0.2))
        except Exception as e:
            last_error = e
            print(f"[confidence] unexpected verbalized error: {type(e).__name__} - {e}")
            break

    print(f"[confidence] verbalized_confidence fallback used: {type(last_error).__name__ if last_error else 'Unknown'}")
    return FALLBACK_VERBALIZED_CONFIDENCE

# -----------------------------------------------------------------------------
# SELF-CONSISTENCY
# -----------------------------------------------------------------------------

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
                    model=SELF_CONSISTENCY_MODEL,
                    messages=[
                        {"role": "system", "content": _CANDIDATE_SYSTEM_PROMPT},
                        {"role": "user", "content": _CANDIDATE_USER_TEMPLATE.format(context=context, query=query)},
                    ],
                    temperature=SELF_CONSISTENCY_TEMPERATURE,
                    max_tokens=150,
                )
            return response.choices[0].message.content
        except RateLimitError as e:
            last_error = e
            if attempt >= max_retries: break
            await asyncio.sleep(0.75 * (attempt + 1))
        except APIStatusError as e:
            last_error = e
            if attempt >= max_retries: break
            await asyncio.sleep(0.5 * (attempt + 1))
        except Exception as e:
            last_error = e
            print(f"[confidence] unexpected candidate error: {type(e).__name__} - {e}")
            break

    print(f"[confidence] candidate answer generation failed: {type(last_error).__name__ if last_error else 'Unknown'}")
    return None

_WORD_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> List[str]:
    stopwords = {"the", "and", "for", "with", "from", "this", "that", "are", "was", "were", "is", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "can", "could", "may", "might", "must", "of", "to", "in", "on", "at", "by", "it", "as", "an", "a", "or", "if", "but", "not", "no", "so", "than", "then"}
    return [t for t in _WORD_RE.findall(text.lower()) if t not in stopwords and len(t) > 2]

def _tf_cosine_similarity(text_a: str, text_b: str) -> float:
    tokens_a = Counter(_tokenize(text_a))
    tokens_b = Counter(_tokenize(text_b))
    if not tokens_a or not tokens_b: return 0.0
    
    shared_vocab = set(tokens_a) | set(tokens_b)
    dot = sum(tokens_a.get(w, 0) * tokens_b.get(w, 0) for w in shared_vocab)
    mag_a = math.sqrt(sum(v * v for v in tokens_a.values()))
    mag_b = math.sqrt(sum(v * v for v in tokens_b.values()))
    if mag_a == 0 or mag_b == 0: return 0.0
    
    return _clamp01(dot / (mag_a * mag_b))

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0: return 0.0
    return _clamp01(dot / (norm_a * norm_b))

async def self_consistency_score(answers: List[Optional[str]], cohere_client=None) -> float:
    valid_answers = [a.strip() for a in answers if a and len(a.strip()) > 20]
    if len(valid_answers) < 2: return FALLBACK_SELF_CONSISTENCY

    if cohere_client is not None:
        try:
            response = await asyncio.to_thread(
                cohere_client.embed,
                texts=valid_answers,
                model="embed-english-light-v3.0",
                input_type="classification",
            )
            embeddings = response.embeddings
            similarities = [_cosine_similarity(embeddings[i], embeddings[j]) for i, j in combinations(range(len(valid_answers)), 2)]
            if similarities: return _clamp01(sum(similarities) / len(similarities))
        except Exception as e:
            print(f"[confidence] embedding self-consistency failed: {e}")

    similarities = [_tf_cosine_similarity(a, b) for a, b in combinations(valid_answers, 2)]
    if not similarities: return FALLBACK_SELF_CONSISTENCY
    return _clamp01(sum(similarities) / len(similarities))

# -----------------------------------------------------------------------------
# SCORE FUSION & ESTIMATOR
# -----------------------------------------------------------------------------

@dataclass
class ConfidenceSignals:
    retrieval_similarity_mean: float
    retrieval_score: float
    top1_similarity: float
    top1_top2_margin: float
    verbalized_confidence: float
    self_consistency: float
    fused_score: float
    retrieval_stats: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

def fuse_scores(verbalized_confidence: float, self_consistency: float, retrieval_score: float) -> float:
    score = (
        FUSION_WEIGHTS["verbalized"] * verbalized_confidence +
        FUSION_WEIGHTS["self_consistency"] * self_consistency +
        FUSION_WEIGHTS["retrieval"] * retrieval_score
    )
    return _clamp01(score)

class ConfidenceEstimator:
    def __init__(self, client: AsyncGroq, cohere_client=None):
        self.client = client
        self.cohere_client = cohere_client
        self.semaphore = asyncio.Semaphore(4)  # Limit concurrent Groq requests

    async def estimate(self, query: str, context: str, distances: List[float]) -> ConfidenceSignals:
        # 1. Compute Retrieval Stats properly
        retrieval_stats = compute_retrieval_stats(distances)
        
        # 2. Fire LLM tasks concurrently
        verbalized_task = get_verbalized_confidence(self.client, query, context, self.semaphore)
        candidate_tasks = [get_candidate_answer(self.client, query, context, self.semaphore) for _ in range(SELF_CONSISTENCY_N)]
        
        results = await asyncio.gather(verbalized_task, *candidate_tasks, return_exceptions=True)
        verbalized_raw, *candidates_raw = results
        
        verbalized_confidence = verbalized_raw if isinstance(verbalized_raw, float) else FALLBACK_VERBALIZED_CONFIDENCE
        candidates = [c if isinstance(c, str) else None for c in candidates_raw]
        
        # 3. Compute Self Consistency
        self_consistency = await self_consistency_score(candidates, cohere_client=self.cohere_client)
        
        # 4. Fuse Scores
        fused_score = fuse_scores(
            verbalized_confidence=verbalized_confidence,
            self_consistency=self_consistency,
            retrieval_score=retrieval_stats.retrieval_score,
        )
        
        return ConfidenceSignals(
            retrieval_similarity_mean=retrieval_stats.mean_similarity,
            retrieval_score=retrieval_stats.retrieval_score,
            top1_similarity=retrieval_stats.top1_similarity,
            top1_top2_margin=retrieval_stats.top1_top2_margin,
            verbalized_confidence=round(verbalized_confidence, 4),
            self_consistency=round(self_consistency, 4),
            fused_score=round(fused_score, 4),
            retrieval_stats=retrieval_stats.as_dict(),
        )