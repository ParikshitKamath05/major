"""
corrective.py
Phase 1 of the Corrective Uncertainty-Triggered RAG layer.

When the confidence estimator returns a LOW fused score, this module runs a
corrective retrieval pipeline instead of naively increasing top_k:

    1. Query Expansion  — a cheap LLM rewrites the query into N variants.
    2. Multi-Query Retrieval — retrieve top candidates for each variant.
    3. Dedup + Merge — collapse duplicates, keep the best distance per chunk.
    4. Neural Rerank — Cohere Rerank reorders candidates by true relevance.

Designed to be decoupled from main.py: you pass in a synchronous `retrieve_fn`
so this module never imports ChromaDB directly (avoids circular imports and
makes it unit-testable).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Tuple

import cohere
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Reuse the cheap/fast model for query expansion.
QUERY_EXPANSION_MODEL = os.getenv(
    "GROQ_QUERY_EXPANSION_MODEL",
    os.getenv("GROQ_VERBALIZED_MODEL", "openai/gpt-oss-20b"),
).strip()

# Cohere neural reranker model.
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0").strip()

N_EXPANDED_QUERIES = int(os.getenv("N_EXPANDED_QUERIES", "3"))
CANDIDATES_PER_QUERY = int(os.getenv("CANDIDATES_PER_QUERY", "15"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "8"))


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

# Signature of the sync retrieval function you pass in from main.py:
#   retrieve_fn(query: str, k: int) -> (documents, metadatas, distances, ids)
RetrieveFn = Callable[[str, int], Tuple[List[str], List[dict], List[float], List[str]]]


@dataclass
class CorrectiveResult:
    documents: List[str]
    metadatas: List[dict]
    distances: List[float]
    chunk_ids: List[str]
    rerank_scores: List[float]
    expanded_queries: List[str]
    total_candidates: int
    method: str  # "corrective_reranked" | "corrective_distance_fallback"

    def as_dict(self) -> Dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# STEP 1 — QUERY EXPANSION
# -----------------------------------------------------------------------------

_EXPANSION_PROMPT = """You are a search query rewriter for a document retrieval system.

Given the user's original question, generate {n} alternative search queries that
would help retrieve relevant passages from a document. The alternatives should:
- Use different wording / synonyms than the original.
- Cover different angles or sub-topics of the question.
- Be concise (each under 20 words).

Return ONLY a valid JSON array of strings. No explanation, no markdown.

ORIGINAL QUESTION:
{query}

Example output format:
["alternative query 1", "alternative query 2", "alternative query 3"]
"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_query_list(raw_text: Optional[str], fallback: str, n: int) -> List[str]:
    """Robustly parse a JSON list of query strings from the LLM output."""
    queries: List[str] = []

    if raw_text:
        raw_text = raw_text.strip()

        # Try direct JSON parse first.
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                queries = [str(q).strip() for q in data if str(q).strip()]
        except Exception:
            # Try to find an embedded JSON array.
            match = _JSON_ARRAY_RE.search(raw_text)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, list):
                        queries = [str(q).strip() for q in data if str(q).strip()]
                except Exception:
                    pass

    # Always keep the original query first, then append unique expansions.
    result = [fallback]
    for q in queries:
        if q not in result:
            result.append(q)
        if len(result) >= n + 1:
            break

    return result


async def expand_query(
    client: AsyncGroq,
    query: str,
    n: int = N_EXPANDED_QUERIES,
) -> List[str]:
    """
    Generate N alternative query formulations. Always returns at least the
    original query, so retrieval still works if the LLM call fails.
    """
    prompt = _EXPANSION_PROMPT.format(n=n, query=query)

    try:
        response = await client.chat.completions.create(
            model=QUERY_EXPANSION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        raw = response.choices[0].message.content
        return _parse_query_list(raw, fallback=query, n=n)
    except Exception as e:  # noqa: BLE001
        print(f"[corrective] query expansion failed, using original only: {type(e).__name__} - {e}")
        return [query]


# -----------------------------------------------------------------------------
# STEP 2 + 3 — MULTI-QUERY RETRIEVAL + DEDUP/MERGE
# -----------------------------------------------------------------------------

def _merge_candidates(
    per_query_results: List[Tuple[str, List[str], List[dict], List[float], List[str]]],
) -> Tuple[List[str], List[dict], List[float], List[str]]:
    """
    Merge results from multiple queries, deduplicating by chunk_id and keeping
    the BEST (lowest) distance for each unique chunk.
    """
    best: Dict[str, Dict] = {}

    for source_query, documents, metadatas, distances, ids in per_query_results:
        for doc, meta, dist, cid in zip(documents, metadatas, distances, ids):
            if cid not in best or dist < best[cid]["distance"]:
                best[cid] = {
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "chunk_id": cid,
                }

    # Sort by distance ascending (best first) for a stable pre-rerank order.
    merged = sorted(best.values(), key=lambda x: x["distance"])

    documents = [m["document"] for m in merged]
    metadatas = [m["metadata"] for m in merged]
    distances = [m["distance"] for m in merged]
    ids = [m["chunk_id"] for m in merged]

    return documents, metadatas, distances, ids


async def multi_query_retrieve(
    retrieve_fn: RetrieveFn,
    queries: List[str],
    candidates_per_query: int = CANDIDATES_PER_QUERY,
) -> Tuple[List[str], List[dict], List[float], List[str]]:
    """
    Run retrieval for each query (off the event loop) and merge + dedup results.
    """
    per_query_results: List[Tuple[str, List[str], List[dict], List[float], List[str]]] = []

    for q in queries:
        try:
            documents, metadatas, distances, ids = await asyncio.to_thread(
                retrieve_fn, q, candidates_per_query
            )
            per_query_results.append((q, documents, metadatas, distances, ids))
        except Exception as e:  # noqa: BLE001
            print(f"[corrective] retrieval failed for query variant: {type(e).__name__} - {e}")

    return _merge_candidates(per_query_results)


# -----------------------------------------------------------------------------
# STEP 4 — NEURAL RERANK (COHERE)
# -----------------------------------------------------------------------------

def _rerank_sync(
    cohere_client: cohere.Client,
    query: str,
    documents: List[str],
    top_n: int,
) -> Optional[List[Tuple[int, float]]]:
    """
    Synchronous Cohere rerank call. Returns list of (original_index, score)
    sorted by relevance, or None on failure.
    """
    if not documents:
        return None

    try:
        response = cohere_client.rerank(
            query=query,
            documents=documents,
            model=RERANK_MODEL,
            top_n=min(top_n, len(documents)),
        )

        ranked = [
            (result.index, float(result.relevance_score))
            for result in response.results
        ]
        return ranked
    except Exception as e:  # noqa: BLE001
        print(f"[corrective] Cohere rerank failed: {type(e).__name__} - {e}")
        return None


# -----------------------------------------------------------------------------
# ORCHESTRATOR
# -----------------------------------------------------------------------------

class CorrectiveRetriever:
    """
    Orchestrates the full corrective retrieval pipeline.

    Usage (from main.py):
        corrective_retriever = CorrectiveRetriever(async_groq_client, cohere_client)

        result = await corrective_retriever.run(
            query=request.question,
            retrieve_fn=lambda q, k: _retrieve_for_uncertainty(
                request.doc_id, q, k, actual_count
            ),
        )
    """

    def __init__(
        self,
        groq_client: AsyncGroq,
        cohere_client: cohere.Client,
        n_expanded_queries: int = N_EXPANDED_QUERIES,
        candidates_per_query: int = CANDIDATES_PER_QUERY,
        final_top_n: int = FINAL_TOP_N,
    ):
        self.groq_client = groq_client
        self.cohere_client = cohere_client
        self.n_expanded_queries = n_expanded_queries
        self.candidates_per_query = candidates_per_query
        self.final_top_n = final_top_n

    async def run(
        self,
        query: str,
        retrieve_fn: RetrieveFn,
    ) -> CorrectiveResult:
        # Step 1 — expand the query into variants.
        expanded_queries = await expand_query(
            self.groq_client, query, n=self.n_expanded_queries
        )
        print(f"[corrective] expanded queries: {expanded_queries}")

        # Step 2 + 3 — retrieve per variant, merge + dedup.
        documents, metadatas, distances, ids = await multi_query_retrieve(
            retrieve_fn, expanded_queries, self.candidates_per_query
        )

        total_candidates = len(documents)

        if total_candidates == 0:
            return CorrectiveResult(
                documents=[], metadatas=[], distances=[], chunk_ids=[],
                rerank_scores=[], expanded_queries=expanded_queries,
                total_candidates=0, method="no_candidates",
            )

        # Step 4 — neural rerank.
        ranked = await asyncio.to_thread(
            _rerank_sync,
            self.cohere_client,
            query,
            documents,
            self.final_top_n,
        )

        if ranked is not None:
            # Reorder using rerank results.
            final_documents = [documents[i] for i, _ in ranked]
            final_metadatas = [metadatas[i] for i, _ in ranked]
            final_distances = [distances[i] for i, _ in ranked]
            final_ids = [ids[i] for i, _ in ranked]
            final_scores = [score for _, score in ranked]
            method = "corrective_reranked"
        else:
            # Fallback: distance-based ordering (already sorted ascending).
            final_documents = documents[: self.final_top_n]
            final_metadatas = metadatas[: self.final_top_n]
            final_distances = distances[: self.final_top_n]
            final_ids = ids[: self.final_top_n]
            final_scores = [0.0] * len(final_documents)
            method = "corrective_distance_fallback"

        print(
            f"[corrective] {method}: {total_candidates} candidates -> "
            f"{len(final_documents)} final chunks"
        )

        return CorrectiveResult(
            documents=final_documents,
            metadatas=final_metadatas,
            distances=final_distances,
            chunk_ids=final_ids,
            rerank_scores=[round(s, 4) for s in final_scores],
            expanded_queries=expanded_queries,
            total_candidates=total_candidates,
            method=method,
        )