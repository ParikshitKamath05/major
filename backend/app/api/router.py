import asyncio
import datetime
import os
import shutil
import tempfile
import time
import traceback
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import (
    COHERE_EMBEDDING_MODEL,
    GROQ_GENERATION_MODEL,
    GROQ_VISION_MODEL,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_IMAGES_PER_DOCUMENT,
    MAX_UPLOAD_SIZE_MB,
)
from app.core.database import (
    collection,
    confidence_estimator,
    conformal_predictor,
    corrective_retriever,
)
from app.core.logging import log_uncertainty_event
from app.models.schemas import (
    ConfidenceSignalsResponse,
    ConformalFeedbackRequest,
    ConformalFeedbackResponse,
    CorrectiveMetadata,
    DeleteResponse,
    MultiQueryRequest,
    QueryRequest,
    QueryResponse,
    UncertaintyQueryRequest,
    UncertaintyQueryResponse,
    UploadResponse,
)
from app.services.chunking_service import chunk_text
from app.services.pdf_service import (
    extract_text_from_pdf_path,
    process_pdf_images,
    validate_pdf_file,
)
from app.services.rag_service import (
    add_chunks_in_batches,
    build_confidence_response,
    build_context,
    build_sources,
    first_result_lists,
    generate_answer_async,
    query_collection,
    retrieve_for_uncertainty,
)

router = APIRouter()


@router.get("/health")
def health():
    # Include conformal calibration status in health check
    cal_stats = conformal_predictor.get_calibration_stats()

    return {
        "status": "healthy",
        "documents_indexed": collection.count(),
        "generation_model": GROQ_GENERATION_MODEL,
        "vision_model": GROQ_VISION_MODEL,
        "embedding_model": COHERE_EMBEDDING_MODEL,
        "conformal_calibrated": cal_stats.get("is_sufficient", False),
        "conformal_calibration_size": cal_stats.get("size", 0),
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document.

    Validates file magic bytes, extracts text and images, chunks content,
    generates embeddings, and stores chunks in ChromaDB.
    """
    original_filename = file.filename or "document.pdf"
    file_bytes = await file.read()

    # File magic byte and size validation
    validate_pdf_file(file_bytes, original_filename)

    if len(file_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE_MB}MB",
        )

    temp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        # --- Text extraction ---
        text = extract_text_from_pdf_path(temp_path)

        # --- Image processing ---
        image_descriptions = process_pdf_images(temp_path, MAX_IMAGES_PER_DOCUMENT)

        # --- Combine text + image descriptions ---
        full_text = text.strip()
        if image_descriptions:
            if full_text:
                full_text += "\n\n"
            full_text += "\n\n".join(image_descriptions)

        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text or image descriptions could be extracted from the PDF",
            )

        chunks = chunk_text(full_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short to process")

        doc_id = uuid.uuid4().hex[:12]
        indexed_chunks = add_chunks_in_batches(doc_id, original_filename, chunks)

        return UploadResponse(
            doc_id=doc_id,
            filename=original_filename,
            chunks=indexed_chunks,
            message=f"Successfully indexed {indexed_chunks} chunks",
        )

    except HTTPException:
        raise
    except Exception as e:
        print("=" * 50)
        print("UPLOAD ERROR:")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query one document using standard RAG.
    """
    existing = collection.get(where={"doc_id": request.doc_id})
    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found",
        )

    actual_count = len(existing["ids"])
    results = query_collection(
        question=request.question,
        n_results=min(request.top_k, actual_count),
        where={"doc_id": request.doc_id},
    )

    documents, metadatas, _, _ = first_result_lists(results)
    if not documents:
        return QueryResponse(
            answer="No relevant content found in the document.",
            sources=[],
            doc_id=request.doc_id,
        )

    context = build_context(documents, metadatas)
    answer = await generate_answer_async(request.question, context, num_docs=1)
    sources = build_sources(documents, metadatas)

    return QueryResponse(
        answer=answer,
        sources=sources,
        doc_id=request.doc_id,
    )


@router.post("/query/uncertainty", response_model=UncertaintyQueryResponse)
async def query_with_uncertainty(request: UncertaintyQueryRequest):
    """
    Uncertainty-Triggered RAG with Semantic Entropy, EDL Fusion,
    Conformal Prediction, and Corrective Escalation.

    Decision pipeline:
        1. Retrieve documents and compute retrieval stats
        2. Sample N candidate answers → semantic entropy + self-consistency
        3. Fuse via Evidential Deep Learning (Dirichlet)
        4. Apply conformal prediction for calibrated decision
        5. High confidence → answer directly
        6. Low confidence → answer with warning
        7. Abstain → corrective retrieval escalation
    """
    request_id = uuid.uuid4().hex
    start_time = time.perf_counter()

    existing = await asyncio.to_thread(
        collection.get,
        where={"doc_id": request.doc_id},
    )

    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found",
        )

    actual_count = len(existing["ids"])

    documents, metadatas, distances, chunk_ids = await asyncio.to_thread(
        retrieve_for_uncertainty,
        request.doc_id,
        request.question,
        request.top_k,
        actual_count,
    )

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the document",
        )

    context = build_context(documents, metadatas)

    # ── Uncertainty Estimation (Semantic Entropy + EDL) ──
    signals = await confidence_estimator.estimate(
        request.question,
        context,
        distances,
    )

    # ── Conformal Prediction (replaces static thresholds) ──
    predictive_confidence = float(
        getattr(signals, "predictive_confidence", 0.0)
    )
    conformal_decision = conformal_predictor.predict(predictive_confidence)
    conformal_thresholds = conformal_decision.thresholds

    escalated = False
    abstained = False
    warning: Optional[str] = None
    corrective_meta: Optional[CorrectiveMetadata] = None
    final_answer = ""
    decision = conformal_decision.decision

    if decision == "high_confidence":
        final_answer = await generate_answer_async(request.question, context, 1)

    elif decision == "low_confidence":
        warning = "low_confidence"
        final_answer = await generate_answer_async(request.question, context, 1)

    else:
        # decision == "abstain" → escalate with corrective retrieval
        decision = "escalated_retrieval_triggered"
        warning = "escalated_retrieval_triggered"
        escalated = True

        def sync_retrieve(q: str, k: int):
            return retrieve_for_uncertainty(request.doc_id, q, k, actual_count)

        corrective_result = await corrective_retriever.run(
            query=request.question,
            retrieve_fn=sync_retrieve,
        )

        if corrective_result.documents:
            documents = corrective_result.documents
            metadatas = corrective_result.metadatas
            chunk_ids = corrective_result.chunk_ids
            context = build_context(documents, metadatas)

            corrective_meta = CorrectiveMetadata(
                expanded_queries=corrective_result.expanded_queries,
                total_candidates=corrective_result.total_candidates,
                final_chunks=len(corrective_result.documents),
                rerank_scores=corrective_result.rerank_scores,
                method=corrective_result.method,
            )

            max_rerank_score = (
                max(corrective_result.rerank_scores)
                if corrective_result.rerank_scores
                else 0.0
            )

            if max_rerank_score < 0.25:
                abstained = True
                warning = "insufficient_context_after_escalation"
                decision = "abstained_after_escalation"
                final_answer = (
                    "I cannot answer this question reliably. The provided "
                    "documents do not contain sufficient information "
                    "regarding this specific query."
                )
            else:
                final_answer = await generate_answer_async(
                    request.question, context, 1
                )
        else:
            abstained = True
            warning = "escalation_failed"
            decision = "abstained_after_escalation"
            final_answer = (
                "I cannot answer this question reliably based on the "
                "provided documents."
            )

    # ── Auto-calibration: record proxy label for conformal ──
    # Proxy heuristic: if we answered without abstaining, treat as
    # "correct" (optimistic). Real feedback via /query/feedback endpoint
    # overrides this.
    if not abstained:
        conformal_predictor.add_calibration_point(
            confidence=predictive_confidence,
            is_correct=True,
            request_id=request_id,
        )

    sources = build_sources(documents, metadatas)
    latency_ms = (time.perf_counter() - start_time) * 1000

    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_id": request_id,
        "query": request.question,
        "doc_id": request.doc_id,
        "initial_top_k": request.top_k,
        "escalated": escalated,
        "abstained": abstained,
        "signals": signals.as_dict() if hasattr(signals, "as_dict") else vars(signals),
        "decision": decision,
        "warning": warning,
        "retrieved_chunk_ids": chunk_ids,
        "latency_ms": latency_ms,
        "conformal": conformal_decision.as_dict(),
        "thresholds": {
            "conformal_abstention": conformal_thresholds.abstention_threshold,
            "conformal_high": conformal_thresholds.high_confidence_threshold,
            "is_calibrated": conformal_thresholds.is_calibrated,
        },
        "models": {
            "generation": GROQ_GENERATION_MODEL,
            "vision": GROQ_VISION_MODEL,
            "embedding": COHERE_EMBEDDING_MODEL,
        },
    }

    if corrective_meta:
        log_entry["corrective_metadata"] = corrective_meta.model_dump()

    log_uncertainty_event(log_entry)

    return UncertaintyQueryResponse(
        request_id=request_id,
        answer=final_answer,
        sources=sources,
        doc_id=request.doc_id,
        confidence=build_confidence_response(signals, conformal_thresholds),
        warning=warning,
        escalated=escalated,
        abstained=abstained,
        decision=decision,
        corrective_metadata=corrective_meta,
        retrieved_chunk_ids=chunk_ids,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# CONFORMAL FEEDBACK ENDPOINT
# ---------------------------------------------------------------------------

@router.post("/query/feedback", response_model=ConformalFeedbackResponse)
async def submit_feedback(request: ConformalFeedbackRequest):
    """
    Submit ground-truth feedback for a past prediction.

    This updates the conformal prediction calibration store, improving
    the adaptive thresholds over time. Without this, the system uses
    optimistic proxy labels (non-abstained → correct).

    Once sufficient calibration data is collected (≥ min_calibration_size),
    the conformal predictor switches from static fallback thresholds to
    statistically calibrated adaptive thresholds with coverage guarantees.
    """
    confidence = request.confidence
    if confidence is None:
        # Default: use a neutral value; the request_id helps trace the real one
        confidence = 0.5

    conformal_predictor.add_calibration_point(
        confidence=confidence,
        is_correct=request.is_correct,
        request_id=request.request_id,
    )

    stats = conformal_predictor.get_calibration_stats()

    return ConformalFeedbackResponse(
        message=(
            f"Calibration point recorded (is_correct={request.is_correct}). "
            f"Calibration {'active' if stats['is_sufficient'] else 'bootstrapping'}."
        ),
        calibration_size=stats["size"],
    )


# ---------------------------------------------------------------------------
# CONFORMAL STATUS ENDPOINT
# ---------------------------------------------------------------------------

@router.get("/conformal/status")
def conformal_status():
    """Return the current conformal prediction status and thresholds."""
    thresholds = conformal_predictor.get_thresholds()
    stats = conformal_predictor.get_calibration_stats()

    return {
        "thresholds": thresholds.as_dict(),
        "calibration": stats,
    }


@router.post("/multi-query", response_model=QueryResponse)
async def multi_query_documents(request: MultiQueryRequest):
    """
    Query across multiple documents.
    """
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="No documents available")

    documents: List[str] = []
    metadatas: List[dict] = []

    if request.doc_ids:
        found_any_document = False

        for doc_id in request.doc_ids:
            existing = collection.get(where={"doc_id": doc_id})
            if not existing["ids"]:
                continue

            found_any_document = True
            results = query_collection(
                question=request.question,
                n_results=min(request.top_k, len(existing["ids"])),
                where={"doc_id": doc_id},
            )

            docs, metas, _, _ = first_result_lists(results)
            documents.extend(docs)
            metadatas.extend(metas)

        if not found_any_document:
            raise HTTPException(
                status_code=404,
                detail="No documents found for selected IDs",
            )

    else:
        results = query_collection(
            question=request.question,
            n_results=min(40, collection.count()),
        )
        documents, metadatas, _, _ = first_result_lists(results)

    if not documents:
        return QueryResponse(
            answer="No relevant content found in the selected documents.",
            sources=[],
            doc_id=None,
        )

    context = build_context(documents, metadatas)
    num_docs = len(
        {
            (metadata or {}).get("doc_id")
            for metadata in metadatas
            if (metadata or {}).get("doc_id")
        }
    )

    answer = await generate_answer_async(
        request.question,
        context,
        num_docs=max(1, num_docs),
    )

    sources = build_sources(documents, metadatas)

    return QueryResponse(
        answer=answer,
        sources=sources,
        doc_id=None,
    )


@router.get("/docs")
def list_documents():
    """
    List all uploaded documents with their chunk counts.
    """
    if collection.count() == 0:
        return {"documents": [], "total": 0}

    all_data = collection.get()
    docs = {}

    for metadata in all_data.get("metadatas", []):
        metadata = metadata or {}
        doc_id = metadata.get("doc_id")

        if not doc_id:
            continue

        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": metadata.get("filename", "unknown"),
                "chunks": 0,
            }

        docs[doc_id]["chunks"] += 1

    return {
        "documents": list(docs.values()),
        "total": len(docs),
    }


@router.delete("/delete/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from the vector database.
    """
    existing = collection.get(where={"doc_id": doc_id})

    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found",
        )

    collection.delete(ids=existing["ids"])

    return DeleteResponse(
        message=f"Document '{doc_id}' deleted successfully",
        deleted_chunks=len(existing["ids"]),
    )


@router.delete("/clear-all")
def clear_all_documents():
    """
    Delete ALL documents from the database.
    """
    count = collection.count()

    if count > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)

    return {
        "message": f"Cleared all {count} chunks from database",
    }
