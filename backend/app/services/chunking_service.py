from typing import List
from app.core.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP


def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks, prioritizing paragraph and line breaks.
    """
    text = (text or "").strip()

    if not text:
        return []

    max_chars = max(100, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            paragraph_pos = text.rfind("\n\n", start, end)
            newline_pos = text.rfind("\n", start, end)
            space_pos = text.rfind(" ", start, end)

            # Prioritize breaking at paragraph, then newline, then space
            break_pos = -1
            for pos in (paragraph_pos, newline_pos, space_pos):
                if pos > start + (max_chars // 2):
                    break_pos = pos
                    break

            if break_pos > start:
                end = break_pos

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        new_start = max(0, end - overlap)

        if new_start <= start:
            new_start = end

        start = new_start

    return chunks
