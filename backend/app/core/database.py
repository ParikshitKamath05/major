from typing import List
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import AsyncGroq, Groq

from app.core.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    COHERE_API_KEY,
    COHERE_EMBEDDING_MODEL,
    CONFORMAL_ALPHA,
    CONFORMAL_CALIBRATION_PATH,
    CONFORMAL_DELTA,
    CONFORMAL_MIN_CALIBRATION_SIZE,
    GROQ_API_KEY,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from Confidence import ConfidenceEstimator
from corrective import CorrectiveRetriever
from conformal import ConformalPredictor

# -----------------------------------------------------------------------------
# CLIENT INITIALIZATION
# -----------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# Branch B: Cohere quota exhausted — cohere_client set to None.
# Confidence.py handles None gracefully (falls back to TF-IDF similarity).
cohere_client = None
groq_client = Groq(api_key=GROQ_API_KEY)
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
corrective_retriever = CorrectiveRetriever(async_groq_client, cohere_client)


# Branch B: local all-MiniLM-L6-v2 embeddings (384-dim) — matches stored vectors.
# CustomCohereEmbedding removed; Cohere quota exhausted.
embedding_fn = DefaultEmbeddingFunction()


collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

try:
    confidence_estimator = ConfidenceEstimator(
        async_groq_client,
        cohere_client=cohere_client,
    )
except TypeError:
    confidence_estimator = ConfidenceEstimator(async_groq_client)

# -----------------------------------------------------------------------------
# CONFORMAL PREDICTOR
# -----------------------------------------------------------------------------

conformal_predictor = ConformalPredictor(
    alpha=CONFORMAL_ALPHA,
    delta=CONFORMAL_DELTA,
    calibration_path=CONFORMAL_CALIBRATION_PATH,
    min_calibration_size=CONFORMAL_MIN_CALIBRATION_SIZE,
    fallback_low=LOW_CONFIDENCE_THRESHOLD,
    fallback_high=HIGH_CONFIDENCE_THRESHOLD,
)
