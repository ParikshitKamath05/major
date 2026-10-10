from typing import List
import chromadb
import cohere
from chromadb.api.types import EmbeddingFunction, Embeddings
from groq import AsyncGroq, Groq

from app.core.config import (
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
cohere_client = cohere.Client(COHERE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
corrective_retriever = CorrectiveRetriever(async_groq_client, cohere_client)


class CustomCohereEmbedding(EmbeddingFunction):
    """
    Cohere embedding function for ChromaDB.

    Documents are embedded with input_type="search_document".
    Queries should be embedded with input_type="search_query" using embed_queries().
    """

    def __init__(self):
        self.client = cohere_client
        self.model = COHERE_EMBEDDING_MODEL

    def _embed(self, input: List[str], input_type: str) -> Embeddings:
        if not input:
            return []

        response = self.client.embed(
            texts=input,
            model=self.model,
            input_type=input_type,
        )
        return response.embeddings

    def __call__(self, input: List[str]) -> Embeddings:
        return self._embed(input, "search_document")

    def embed_queries(self, input: List[str]) -> Embeddings:
        return self._embed(input, "search_query")


embedding_fn = CustomCohereEmbedding()

collection = chroma_client.get_or_create_collection(
    name="documents",
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
