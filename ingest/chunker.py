"""Simple word-based chunker with overlap. Good enough for financial text/tables."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str          # file name
    location: str        # page number / sheet name / section label
    doc_date: str = ""   # e.g. "2024-Q2", "FY2023" - helps the LLM reason about recency
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, source: str, location: str, doc_date: str = "",
                chunk_size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping chunks of ~chunk_size words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        piece = " ".join(words[start:end])
        if piece.strip():
            chunks.append(Chunk(text=piece, source=source, location=location, doc_date=doc_date))
        if end == len(words):
            break
        start = end - overlap
    return chunks
