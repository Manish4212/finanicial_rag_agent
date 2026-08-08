"""
Orchestrates: vector search -> RBAC filter -> feedback-based rerank -> top_k.

Order matters: we filter by role BEFORE truncating to top_k, so a restricted
chunk can never "use up" a slot that a permitted chunk needed, and it never
reaches the LLM even transiently.
"""

import config
from rag.embed_store import query as vector_query
from rag.rbac import filter_chunks_by_role
from feedback.store import get_feedback_boosts


def _chroma_results_to_dicts(results) -> list[dict]:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "location": meta.get("location"),
                "doc_date": meta.get("doc_date"),
                "category": meta.get("category"),
                "flagged_injection": meta.get("flagged_injection", False),
                "distance": dist,
            }
        )
    return out


def retrieve(query_text: str, role: str, top_k: int = None) -> dict:
    top_k = top_k or config.TOP_K

    raw_results = vector_query(query_text, n_results=config.CANDIDATE_K)
    candidates = _chroma_results_to_dicts(raw_results)

    # RBAC enforcement happens here, before anything else touches these chunks.
    allowed, num_blocked = filter_chunks_by_role(candidates, role)

    # Feedback-based reranking: boost chunks that were part of past
    # positively-rated answers to similar queries.
    boosts = get_feedback_boosts(query_text, role)
    for chunk in allowed:
        key = f"{chunk['source']}::{chunk['location']}"
        chunk["score"] = -chunk["distance"] + boosts.get(key, 0.0)

    allowed.sort(key=lambda c: c["score"], reverse=True)
    final_chunks = allowed[:top_k]

    return {
        "chunks": final_chunks,
        "num_candidates": len(candidates),
        "num_blocked_by_rbac": num_blocked,
        "num_returned": len(final_chunks),
    }
