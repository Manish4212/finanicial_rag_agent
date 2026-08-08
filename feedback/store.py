"""
Feedback loop storage and retrieval.

Every answer can be rated thumbs up/down, optionally with a free-text
correction. This module:
  1. Persists feedback events to SQLite.
  2. On future queries, finds past queries that are semantically similar
     (cosine similarity over the same local embedding model used for
     retrieval) and:
       - boosts the chroma-retrieval score of chunks that were part of a
         positively-rated answer to a similar query (see
         get_feedback_boosts, used by rag/retriever.py)
       - surfaces past corrections/positive answers as few-shot examples in
         the prompt (see get_few_shot_examples, used by rag/answer.py)

This is intentionally a simple, explainable mechanism: no fine-tuning, no
hidden state, and every future behavior change is traceable back to a
specific stored feedback row.
"""

import json
import sqlite3
import time
from typing import List, Dict

import numpy as np

import config
from rag.embed_store import embed_texts

FEEDBACK_BOOST = 0.15   # score bump applied to chunks tied to a positive past answer
FEEDBACK_PENALTY = -0.15  # score penalty for chunks tied to a negative past answer


def _connect():
    conn = sqlite3.connect(config.FEEDBACK_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            role TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT NOT NULL,          -- 'up' or 'down'
            correction TEXT,
            chunk_ids TEXT NOT NULL,       -- JSON list of "source::location"
            query_embedding TEXT NOT NULL, -- JSON list of floats
            run_id TEXT,                   -- LangSmith run id for this answer, if tracing was on
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def record_feedback(query: str, role: str, answer: str, rating: str,
                     chunk_ids: List[str], correction: str = "", run_id: str = None):
    assert rating in ("up", "down")
    embedding = embed_texts([query])[0]
    conn = _connect()
    conn.execute(
        "INSERT INTO feedback (query, role, answer, rating, correction, chunk_ids, "
        "query_embedding, run_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            query, role, answer, rating, correction,
            json.dumps(chunk_ids), json.dumps(embedding), run_id, time.time(),
        ),
    )
    conn.commit()
    conn.close()

    if run_id and config.LANGSMITH_ENABLED:
        _push_feedback_to_langsmith(run_id, rating, correction)


def _push_feedback_to_langsmith(run_id: str, rating: str, correction: str = ""):
    """Attach the user's 👍/👎 directly to the matching LangSmith trace, so
    the trace and the human judgment of it live in one place. Fails silently
    (logged, not raised) so a LangSmith hiccup never breaks the feedback UI."""
    try:
        from langsmith import Client
        client = Client(api_key=config.LANGCHAIN_API_KEY)
        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=1.0 if rating == "up" else 0.0,
            comment=correction or None,
        )
    except Exception as e:
        print(f"[langsmith] failed to push feedback for run {run_id}: {e}")


def _all_feedback(role: str = None) -> List[dict]:
    conn = _connect()
    cur = conn.cursor()
    if role:
        cur.execute("SELECT query, role, answer, rating, correction, chunk_ids, "
                     "query_embedding FROM feedback WHERE role = ?", (role,))
    else:
        cur.execute("SELECT query, role, answer, rating, correction, chunk_ids, "
                     "query_embedding FROM feedback")
    rows = cur.fetchall()
    conn.close()

    out = []
    for query, role_, answer, rating, correction, chunk_ids_json, emb_json in rows:
        out.append({
            "query": query,
            "role": role_,
            "answer": answer,
            "rating": rating,
            "correction": correction or "",
            "chunk_ids": json.loads(chunk_ids_json),
            "embedding": np.array(json.loads(emb_json)),
        })
    return out


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _similar_past_feedback(query: str, role: str, threshold: float = None) -> List[dict]:
    threshold = threshold or config.FEEDBACK_SIMILARITY_THRESHOLD
    query_emb = np.array(embed_texts([query])[0])

    similar = []
    for fb in _all_feedback(role=role):
        sim = _cosine_sim(query_emb, fb["embedding"])
        if sim >= threshold:
            fb["similarity"] = sim
            similar.append(fb)

    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return similar


def get_feedback_boosts(query: str, role: str) -> Dict[str, float]:
    """Return {chunk_id: score_adjustment} based on similar past feedback."""
    boosts: Dict[str, float] = {}
    for fb in _similar_past_feedback(query, role):
        adjustment = FEEDBACK_BOOST if fb["rating"] == "up" else FEEDBACK_PENALTY
        for chunk_id in fb["chunk_ids"]:
            boosts[chunk_id] = boosts.get(chunk_id, 0.0) + adjustment
    return boosts


def get_few_shot_examples(query: str, role: str, max_examples: int = 3) -> List[dict]:
    """Return past positively-rated answers or corrections for similar queries,
    to inject into the prompt as few-shot guidance."""
    examples = []
    for fb in _similar_past_feedback(query, role)[:max_examples]:
        if fb["rating"] == "down" and fb["correction"]:
            examples.append({"type": "correction", "query": fb["query"], "correction": fb["correction"]})
        elif fb["rating"] == "up":
            examples.append({"type": "positive", "query": fb["query"], "answer": fb["answer"]})
    return examples


def get_all_feedback_for_display() -> List[dict]:
    """Used by the UI to show a feedback history / stats panel."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT query, role, rating, correction, created_at FROM feedback ORDER BY created_at DESC LIMIT 50"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"query": q, "role": r, "rating": rating, "correction": c, "created_at": ts}
        for q, r, rating, c, ts in rows
    ]
