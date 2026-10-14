import os
import time
from typing import List, Optional, Tuple
from fastapi import HTTPException

from app.core.config import GROQ_GENERATION_MODEL, GROQ_MAX_TOKENS
from app.core.database import async_groq_client, collection, embedding_fn
from app.core.prompts import SYSTEM_PROMPT
from app.models.schemas import ConfidenceSignalsResponse, SourceInfo
from Confidence import ConfidenceSignals


def as_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def first_result_lists(results: dict):
    """Safely unpack the first row list from a ChromaDB query result."""
    results = results or {}
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []
    ids = (results.get("ids") or [[]])[0] or []

    return documents, metadatas, distances, ids


def query_collection(
    question: str,
    n_results: int,
    where: Optional[dict] = None,
) -> dict:
    """Query ChromaDB using search_query embeddings."""
    if n_results <= 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

    query_kwargs = {
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    if where:
        query_kwargs["where"] = where

    if hasattr(embedding_fn, "embed_queries"):
        query_kwargs["query_embeddings"] = embedding_fn.embed_queries([question])
    else:
        query_kwargs["query_texts"] = [question]

    return collection.query(**query_kwargs)


def retrieve_for_uncertainty(doc_id: str, question: str, k: int, actual_count: int) -> Tuple[List[str], List[dict], List[float], List[str]]:
    """Sync ChromaDB query including distances and IDs for uncertainty estimation."""
    results = collection.query(
        query_texts=[question],
        n_results=min(k, actual_count),
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]
    return documents, metadatas, distances, ids


def add_chunks_in_batches(doc_id: str, filename: str, chunks: List[str]) -> int:
    """Add chunks to ChromaDB in smaller batches."""
    if not chunks:
        return 0

    batch_size = int(os.getenv("CHROMA_ADD_BATCH_SIZE", "64"))
    if batch_size <= 0:
        batch_size = 64

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    return len(chunks)


def build_sources(documents: List[str], metadatas: List[dict]) -> List[SourceInfo]:
    sources: List[SourceInfo] = []

    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        preview = document[:200] + "..." if len(document) > 200 else document

        sources.append(
            SourceInfo(
                filename=str(metadata.get("filename", "unknown")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                text_preview=preview,
            )
        )

    return sources


def build_context(documents: List[str], metadatas: List[dict]) -> str:
    context_parts: List[str] = []

    for i, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}
        filename = metadata.get("filename", "unknown")
        chunk_index = metadata.get("chunk_index", i - 1)
        page = metadata.get("page")
        page_part = f" | Page: {page}" if page is not None else ""

        context_parts.append(
            f"[Chunk {i} | File: {filename} | Chunk Index: {chunk_index}{page_part}]: {document}"
        )

    return "\n\n".join(context_parts)


async def generate_answer_async(query: str, context: str, num_docs: int = 1) -> str:
    """Call Groq LLM asynchronously with RAG context."""
    if not context.strip():
        return "The provided documents do not contain enough information to answer this query."

    user_content = f"""DOCUMENTS ANALYZED: {num_docs}
NUMBER OF CHUNKS PROVIDED: {context.count("[Chunk")}

DOCUMENT CHUNKS:
{context}

USER QUESTION: {query}

Provide a comprehensive, well-reasoned analysis."""

    last_error: Optional[Exception] = None

    for attempt in range(3):
        try:
            response = await async_groq_client.chat.completions.create(
                model=GROQ_GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=GROQ_MAX_TOKENS,
            )

            time.sleep(3.5)  # Groq free tier: ~30 req/min
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))

    raise HTTPException(status_code=500, detail=f"LLM error: {str(last_error)}")


def build_confidence_response(
    signals: ConfidenceSignals,
    conformal_thresholds=None,
) -> ConfidenceSignalsResponse:
    """
    Convert ConfidenceSignals into API response model.

    Includes backward-compatible fields alongside new semantic entropy,
    EDL, and conformal prediction fields.
    """
    data = signals.as_dict() if hasattr(signals, "as_dict") else vars(signals)
    retrieval_mean = as_float(data.get("retrieval_similarity_mean"), 0.0)

    response = ConfidenceSignalsResponse(
        # Retrieval
        retrieval_similarity_mean=retrieval_mean,
        retrieval_score=as_float(data.get("retrieval_score"), retrieval_mean),
        top1_similarity=as_float(data.get("top1_similarity"), retrieval_mean),
        top1_top2_margin=as_float(data.get("top1_top2_margin"), 0.0),
        # Backward compat
        verbalized_confidence=as_float(data.get("verbalized_confidence"), 0.0),
        self_consistency=as_float(data.get("self_consistency"), 0.0),
        fused_score=as_float(data.get("fused_score"), 0.0),
        # Semantic Entropy
        semantic_entropy=as_float(data.get("semantic_entropy"), 0.0),
        normalized_semantic_entropy=as_float(data.get("normalized_semantic_entropy"), 0.0),
        epistemic_uncertainty=as_float(data.get("epistemic_uncertainty"), 0.0),
        aleatoric_uncertainty=as_float(data.get("aleatoric_uncertainty"), 0.0),
        num_semantic_clusters=int(data.get("num_semantic_clusters", 0)),
        semantic_confidence=as_float(data.get("semantic_confidence"), 0.0),
        # EDL
        belief_correct=as_float(data.get("belief_correct"), 0.0),
        belief_incorrect=as_float(data.get("belief_incorrect"), 0.0),
        evidential_uncertainty=as_float(data.get("evidential_uncertainty"), 0.0),
        predictive_confidence=as_float(data.get("predictive_confidence"), 0.0),
        dirichlet_strength=as_float(data.get("dirichlet_strength"), 0.0),
        total_evidence=as_float(data.get("total_evidence"), 0.0),
    )

    # Conformal prediction thresholds (if available)
    if conformal_thresholds is not None:
        ct = conformal_thresholds
        response.conformal_is_calibrated = getattr(ct, "is_calibrated", False)
        response.conformal_calibration_size = getattr(ct, "calibration_size", 0)
        response.conformal_abstention_threshold = as_float(
            getattr(ct, "abstention_threshold", 0.0)
        )
        response.conformal_high_threshold = as_float(
            getattr(ct, "high_confidence_threshold", 0.0)
        )
        response.conformal_coverage_guarantee = as_float(
            getattr(ct, "coverage_guarantee", 0.0)
        )

    return response
