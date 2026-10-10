import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY is not set. Add it to your .env file.")

# File system paths
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR.parent / "chroma_db"))).resolve()
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs"))).resolve()
LOGS_DIR.mkdir(parents=True, exist_ok=True)

UNCERTAINTY_LOG_PATH = LOGS_DIR / "uncertainty_logs.jsonl"

# Model Configurations
GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-light-v3.0")

# Chunking Parameters
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHROMA_ADD_BATCH_SIZE = int(os.getenv("CHROMA_ADD_BATCH_SIZE", "64"))

# Processing Limits
MAX_IMAGES_PER_DOCUMENT = int(os.getenv("MAX_IMAGES_PER_DOCUMENT", "10"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

# ── RAG Thresholds (static fallbacks — used when conformal is uncalibrated) ──
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.75"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.45"))
ESCALATED_TOP_K = int(os.getenv("ESCALATED_TOP_K", "20"))

# ── Conformal Prediction ──
CONFORMAL_ALPHA = float(os.getenv("CONFORMAL_ALPHA", "0.10"))
CONFORMAL_DELTA = float(os.getenv("CONFORMAL_DELTA", "0.05"))
CONFORMAL_CALIBRATION_PATH = str(
    LOGS_DIR / os.getenv("CONFORMAL_CALIBRATION_FILE", "conformal_calibration.jsonl")
)
CONFORMAL_MIN_CALIBRATION_SIZE = int(os.getenv("CONFORMAL_MIN_CALIBRATION_SIZE", "30"))
