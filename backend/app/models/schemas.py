from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class MultiQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceInfo(BaseModel):
    filename: str
    chunk_index: int
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    doc_id: Optional[str] = None


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    message: str


class DeleteResponse(BaseModel):
    message: str
    deleted_chunks: int


class UncertaintyQueryRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class ConfidenceSignalsResponse(BaseModel):
    """
    API response model for the full uncertainty profile.

    Contains backward-compatible fields (verbalized_confidence, fused_score)
    alongside the new semantic entropy, EDL, and conformal prediction fields.
    """

    # ── Retrieval signals ──
    retrieval_similarity_mean: float
    retrieval_score: float = 0.0
    top1_similarity: float = 0.0
    top1_top2_margin: float = 0.0

    # ── Backward-compatible fields ──
    verbalized_confidence: float     # = semantic_confidence (for compat)
    self_consistency: float
    fused_score: float               # = predictive_confidence (for compat)

    # ── Semantic Entropy (Phase 1) ──
    semantic_entropy: float = 0.0
    normalized_semantic_entropy: float = 0.0
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    num_semantic_clusters: int = 0
    semantic_confidence: float = 0.0

    # ── Evidential Deep Learning (Phase 3) ──
    belief_correct: float = 0.0
    belief_incorrect: float = 0.0
    evidential_uncertainty: float = 0.0
    predictive_confidence: float = 0.0
    dirichlet_strength: float = 0.0
    total_evidence: float = 0.0

    # ── Conformal Prediction (Phase 2) ──
    conformal_is_calibrated: bool = False
    conformal_calibration_size: int = 0
    conformal_abstention_threshold: float = 0.0
    conformal_high_threshold: float = 0.0
    conformal_coverage_guarantee: float = 0.0


class ConformalFeedbackRequest(BaseModel):
    """Request to provide ground-truth feedback for conformal calibration."""
    request_id: str = Field(min_length=1)
    is_correct: bool
    confidence: Optional[float] = None


class ConformalFeedbackResponse(BaseModel):
    """Response confirming calibration point was recorded."""
    message: str
    calibration_size: int


class CorrectiveMetadata(BaseModel):
    expanded_queries: List[str] = Field(default_factory=list)
    total_candidates: int = 0
    final_chunks: int = 0
    rerank_scores: List[float] = Field(default_factory=list)
    method: str = "none"


class UncertaintyQueryResponse(BaseModel):
    request_id: Optional[str] = None
    answer: str
    sources: List[SourceInfo]
    doc_id: str
    confidence: ConfidenceSignalsResponse
    warning: Optional[str] = None
    escalated: bool = False
    corrective_metadata: Optional[CorrectiveMetadata] = None
    abstained: bool = False
    decision: Optional[str] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    latency_ms: Optional[float] = None
