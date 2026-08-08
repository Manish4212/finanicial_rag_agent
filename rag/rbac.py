"""
RBAC enforcement at the DATA layer.

This is intentionally the most boring file in the repo: given a role and a
list of retrieved chunks (each carrying a 'category' metadata field set at
ingestion time), it returns only the chunks that role is allowed to see.

Critically, this filtering happens BEFORE any chunk text is placed into an
LLM prompt. The model never sees restricted content, so it cannot leak it -
whether asked directly, asked to "combine" restricted and permitted data, or
asked to work around it via roleplay/injection. There's nothing to leak
because it was never in context.
"""

from config import allowed_categories_for


def filter_chunks_by_role(chunks: list[dict], role: str) -> tuple[list[dict], int]:
    """
    chunks: list of dicts with at least a 'category' key (as produced by
            rag.retriever.RetrievedChunk._asdict() or similar).
    Returns (allowed_chunks, num_blocked).
    """
    allowed = allowed_categories_for(role)
    kept = []
    blocked = 0
    for chunk in chunks:
        if chunk["category"] in allowed:
            kept.append(chunk)
        else:
            blocked += 1
    return kept, blocked
