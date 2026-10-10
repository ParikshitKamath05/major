import json
import logging
from app.core.config import UNCERTAINTY_LOG_PATH

uncertainty_logger = logging.getLogger("documind.uncertainty")
uncertainty_logger.setLevel(logging.INFO)

if not uncertainty_logger.handlers:
    handler = logging.FileHandler(UNCERTAINTY_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    uncertainty_logger.addHandler(handler)

uncertainty_logger.propagate = False


def log_uncertainty_event(entry: dict) -> None:
    """
    Append one JSONL record for /query/uncertainty calls.

    Never raises: logging failures should not break the request.
    """
    try:
        uncertainty_logger.info(json.dumps(entry, ensure_ascii=False, default=str))
    except Exception as e:
        print(f"[uncertainty_logs] failed to write log entry: {e}")
