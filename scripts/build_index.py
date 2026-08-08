"""
Regenerates the full "understanding layer" from data/raw/:
  1. Ingest every .pdf and .xlsx file into chunks.
  2. Classify each chunk into an RBAC category + scan for injection attempts.
  3. Embed chunks locally and upsert into the persistent Chroma index.
  4. Generate a one-paragraph LLM summary per source document and save it to
     data/understanding/<filename>.summary.json - a human/agent-readable
     "understanding file" that can answer high-level questions without a
     full retrieval pass, and gives the walkthrough something concrete to
     point at as "precomputed vs on the fly".

Run:  python scripts/build_index.py
Safe to re-run - embeddings are upserted (dedup by id) and summaries are
overwritten.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ingest.pdf_ingest import ingest_pdf
from ingest.xlsx_ingest import ingest_xlsx
from ingest.classify import classify_chunk, scan_for_injection
from rag.embed_store import add_chunks

os.makedirs(config.UNDERSTANDING_DIR, exist_ok=True)


def _infer_doc_date(filename: str) -> str:
    """Best-effort date/period label pulled from the filename, used as chunk
    metadata so the LLM can reason about recency without re-parsing dates."""
    name = filename.upper()
    for token in name.replace(".", "_").split("_"):
        if token.startswith("FY") or token.startswith("Q1") or token.startswith("Q2") \
           or token.startswith("Q3") or token.startswith("Q4"):
            return token
    return ""


def summarize_document(text_sample: str, filename: str) -> str:
    """One LLM call per document at ingestion time - precomputed once, reused
    for every future query that needs a high-level view of this document."""
    if not config.GROQ_API_KEY:
        return "(summary skipped - GROQ_API_KEY not set)"

    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGroq(
        model=config.GROQ_MODEL,
        groq_api_key=config.GROQ_API_KEY,
        temperature=0.7,
        max_tokens=300,
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke([
                SystemMessage(content="Summarize the following financial document excerpt in 3-4 sentences, "
                                       "focused on what data it contains (e.g. revenue figures, headcount, "
                                       "strategic commentary) so an analyst knows whether to look here for an "
                                       "answer. Do not invent numbers not present in the text."),
                HumanMessage(content=text_sample[:6000]),
            ])
            return response.content or ""
        except Exception as e:
            error_msg = str(e).upper()
            # Catch rate limit errors and wait it out
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    print(f"  ! Rate limit hit for {filename}. Waiting 60 seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(60)
                else:
                    print(f"  ! Failed to summarize {filename} after {max_retries} attempts.")
                    return "(summary generation failed - rate limits exhausted)"
            else:
                # If it's a different error (like Auth or Network), crash normally
                raise e


def process_file(path: str):
    filename = os.path.basename(path)
    doc_date = _infer_doc_date(filename)

    if filename.lower().endswith(".pdf"):
        chunks = ingest_pdf(path, doc_date=doc_date)
    elif filename.lower().endswith(".xlsx"):
        chunks = ingest_xlsx(path, doc_date=doc_date)
    else:
        print(f"Skipping unsupported file: {filename}")
        return

    if not chunks:
        print(f"No content extracted from {filename}")
        return

    categories = [classify_chunk(c.text) for c in chunks]
    injection_flags = [scan_for_injection(c.text) for c in chunks]

    flagged_count = sum(injection_flags)
    if flagged_count:
        print(f"  ! {flagged_count} chunk(s) in {filename} flagged for possible prompt injection")

    add_chunks(chunks, categories, injection_flags)

    # Build the document-level "understanding file"
    full_text_sample = "\n".join(c.text for c in chunks)
    summary = summarize_document(full_text_sample, filename)
    category_breakdown = {}
    for cat in categories:
        category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

    understanding = {
        "source": filename,
        "doc_date": doc_date,
        "num_chunks": len(chunks),
        "category_breakdown": category_breakdown,
        "summary": summary,
    }
    out_path = os.path.join(config.UNDERSTANDING_DIR, f"{filename}.summary.json")
    with open(out_path, "w") as f:
        json.dump(understanding, f, indent=2)

    print(f"Indexed {filename}: {len(chunks)} chunks -> {category_breakdown}")


def main():
    files = [
        os.path.join(config.RAW_DATA_DIR, f)
        for f in sorted(os.listdir(config.RAW_DATA_DIR))
        if f.lower().endswith((".pdf", ".xlsx"))
    ]
    if not files:
        print(f"No .pdf or .xlsx files found in {config.RAW_DATA_DIR}. "
              f"Run scripts/generate_sample_data.py first, or add real files.")
        return

    print(f"Found {len(files)} file(s) to index.\n")
    for path in files:
        process_file(path)
        # Add a buffer delay between files to prevent quick RPM exhaustion
        time.sleep(5)

    print("\nDone. Vector index at:", config.CHROMA_DIR)
    print("Understanding files at:", config.UNDERSTANDING_DIR)


if __name__ == "__main__":
    main()