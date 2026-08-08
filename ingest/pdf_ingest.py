"""Extracts text (and tables where present) from PDFs, page by page."""

import os
import pdfplumber
from ingest.chunker import chunk_text, Chunk


def ingest_pdf(path: str, doc_date: str = "") -> list[Chunk]:
    """Extract per-page text from a PDF and chunk it. Tables are flattened
    into readable text so they chunk/embed well alongside prose."""
    source = os.path.basename(path)
    all_chunks: list[Chunk] = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Flatten any tables on this page into text rows so numbers aren't lost
            tables = page.extract_tables() or []
            table_text_parts = []
            for table in tables:
                for row in table:
                    cells = [str(c).strip() for c in row if c is not None]
                    if cells:
                        table_text_parts.append(" | ".join(cells))
            table_text = "\n".join(table_text_parts)

            full_page_text = (text + "\n" + table_text).strip()
            if not full_page_text:
                continue

            location = f"page {page_num}"
            chunks = chunk_text(full_page_text, source=source, location=location, doc_date=doc_date)
            all_chunks.extend(chunks)

    return all_chunks
