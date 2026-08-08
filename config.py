"""
Central configuration: categories, role permissions, model settings.

This file is the single source of truth for access control. RBAC is enforced
by filtering retrieved chunks against ROLE_PERMISSIONS *before* they ever
reach the LLM prompt (see rag/retriever.py). It is never enforced by asking
the model nicely in a system prompt.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Categories every ingested chunk gets tagged with (see ingest/classify.py)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "revenue",         # revenue, sales, segment/product financials
    "product",         # product lines, units, market info
    "headcount_comp",  # headcount, salaries, compensation, benefits
    "strategy",        # strategic initiatives, M&A, competitive positioning
    "guidance",        # forward-looking guidance / outlook
    "general",         # anything that doesn't fit the above (boilerplate, legal, etc.)
]

# ---------------------------------------------------------------------------
# Role -> set of categories that role is allowed to see.
# "all" is a special value meaning unrestricted access.
# ---------------------------------------------------------------------------
ROLE_PERMISSIONS = {
    "CEO": {"all"},
    "CTO": {"revenue", "product", "strategy", "guidance", "general"},  # no headcount/comp
    "Analyst": {"revenue", "product", "general"},  # no headcount/comp, no strategy, no guidance
}

ROLES = list(ROLE_PERMISSIONS.keys())


def allowed_categories_for(role: str) -> set:
    """Return the set of categories a role may see. Fails closed on unknown roles."""
    perms = ROLE_PERMISSIONS.get(role, set())
    if "all" in perms:
        return set(CATEGORIES)
    return perms


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
UNDERSTANDING_DIR = os.path.join(BASE_DIR, "data", "understanding")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
FEEDBACK_DB_PATH = os.path.join(BASE_DIR, "feedback.sqlite3")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Local embedding model (no API key required)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Groq chat model used for answering + ingestion-time summarization.
# Override with GROQ_MODEL env var if needed.
GROQ_MODEL = os.environ.get("GROQ_MODEL", os.environ.get("GEMINI_MODEL", "llama3-8b-8192"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

# Backward-compatible aliases for older code paths.
GEMINI_MODEL = GROQ_MODEL
GEMINI_API_KEY = GROQ_API_KEY

# ---------------------------------------------------------------------------
# LangSmith (optional tracing/observability for the LangChain LLM calls)
# ---------------------------------------------------------------------------
# If LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set, LangChain
# automatically traces every llm.invoke() call to your LangSmith project -
# no code changes needed beyond these env vars. We additionally capture the
# resulting run_id per query (see rag/answer.py) so that user feedback
# (👍/👎) can be attached directly to the matching LangSmith trace.
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "financial-rag-agent")
LANGSMITH_ENABLED = LANGCHAIN_TRACING_V2.lower() == "true" and bool(LANGCHAIN_API_KEY)

# Retrieval
TOP_K = 6              # chunks returned to the LLM after RBAC filtering
CANDIDATE_K = 25        # chunks pulled from the vector store before filtering/reranking

# Feedback-based reranking
FEEDBACK_SIMILARITY_THRESHOLD = 0.85  # cosine sim to consider a past query "similar"

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
# How many past (query, answer) turns to replay into the prompt so the
# assistant can handle follow-up questions. Kept small on purpose: every
# past turn costs prompt tokens, and memory is bucketed per-role (see
# memory/conversation.py) so it never mixes RBAC contexts.
MEMORY_MAX_TURNS = 6

# ---------------------------------------------------------------------------
# Tools (function/tool calling)
# ---------------------------------------------------------------------------
# Frankfurter is a free FX-rate API, no key required.
CURRENCY_API_BASE_URL = os.environ.get("CURRENCY_API_BASE_URL", "https://api.frankfurter.dev")
CURRENCY_API_TIMEOUT_SECONDS = 10
