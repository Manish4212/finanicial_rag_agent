"""
rag: retrieval-augmented generation layer.

Modules:
    embed_store       - local sentence-transformers embeddings + persistent Chroma vector store
    rbac              - role-based access control filter (the core enforcement point)
    retriever         - orchestrates vector search -> RBAC filter -> feedback rerank
    injection_guard   - prompt-injection defenses applied at query time
    answer            - assembles the final prompt and calls the Groq API

Import from the submodules directly, e.g.:
    from rag.answer import answer_question
    from rag.retriever import retrieve
"""

from rag.rbac import filter_chunks_by_role
from rag.retriever import retrieve
from rag.answer import answer_question

__all__ = ["filter_chunks_by_role", "retrieve", "answer_question"]
