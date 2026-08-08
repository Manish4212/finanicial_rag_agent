"""
Converts spreadsheet rows into sentence-like text before chunking/embedding.

Design choice: LLMs retrieve and reason over prose far better than raw grid
cells. Instead of dumping "Q1 2024 | iPhone | 45000" we turn each row into
something like: "In Q1 2024, iPhone revenue was $45,000 million." This makes
embeddings more semantically meaningful and lets the LLM answer directly
from a retrieved chunk without re-deriving what the columns mean.
"""

import os
import pandas as pd
from ingest.chunker import Chunk


def _row_to_sentence(sheet_name: str, row: pd.Series, columns: list[str]) -> str:
    parts = []
    for col in columns:
        val = row[col]
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")
    return f"[{sheet_name}] " + "; ".join(parts)


def ingest_xlsx(path: str, doc_date: str = "") -> list[Chunk]:
    source = os.path.basename(path)
    all_chunks: list[Chunk] = []

    xls = pd.ExcelFile(path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if df.empty:
            continue
        columns = list(df.columns)

        for idx, row in df.iterrows():
            sentence = _row_to_sentence(sheet_name, row, columns)
            if not sentence.strip():
                continue
            all_chunks.append(
                Chunk(
                    text=sentence,
                    source=source,
                    location=f"sheet '{sheet_name}' row {idx + 2}",  # +2: header + 1-index
                    doc_date=doc_date,
                )
            )
    return all_chunks
