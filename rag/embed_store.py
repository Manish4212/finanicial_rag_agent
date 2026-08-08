"""
Wraps a local sentence-transformers model + a persistent Chroma collection.

Embeddings run locally (no API key needed for this step) so ingestion works
even before you've set up a Groq key - only summarization and
query-time answering need GROQ_API_KEY.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config

_model = None
_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name="financial_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(texts: list[str]):
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def add_chunks(chunks: list, categories: list[str], injection_flags: list[bool]):
    """chunks: list[ingest.chunker.Chunk]"""
    if not chunks:
        return

    collection = get_collection()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids = [f"{c.source}::{c.location}::{i}" for i, c in enumerate(chunks)]
    metadatas = [
        {
            "source": c.source,
            "location": c.location,
            "doc_date": c.doc_date,
            "category": cat,
            "flagged_injection": bool(flag),
        }
        for c, cat, flag in zip(chunks, categories, injection_flags)
    ]

    # Chroma upsert avoids duplicate errors on re-ingestion
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query(query_text: str, n_results: int):
    collection = get_collection()
    query_embedding = embed_texts([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results
