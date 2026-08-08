"""
ingest: turns raw PDF/Excel files into classified, chunked text ready for embedding.

Modules:
    chunker    - word-based chunking with overlap (Chunk dataclass lives here)
    classify   - RBAC category tagging + prompt-injection pattern scanning
    pdf_ingest - PDF -> chunks (text + flattened tables)
    xlsx_ingest- Excel rows -> sentence-like chunks

Import from the submodules directly, e.g.:
    from ingest.pdf_ingest import ingest_pdf
    from ingest.xlsx_ingest import ingest_xlsx
"""

from ingest.chunker import Chunk, chunk_text
from ingest.classify import classify_chunk, scan_for_injection
from ingest.pdf_ingest import ingest_pdf
from ingest.xlsx_ingest import ingest_xlsx

__all__ = [
    "Chunk",
    "chunk_text",
    "classify_chunk",
    "scan_for_injection",
    "ingest_pdf",
    "ingest_xlsx",
]
