# Financial Data Assistant — Agentic RAG with RBAC + Feedback Loop

A working, end-to-end system: ingest financial PDFs/Excel → build an understanding
layer → answer natural-language questions → enforce role-based access control
at the data layer → learn from user feedback.

Built with synthetic Apple-style sample data by default so it runs immediately
with zero external downloads. Swap in real filings any time (see below).

## Architecture at a glance

```
data/raw/*.pdf, *.xlsx
        │
        ▼
  ingest/  (pdf_ingest.py, xlsx_ingest.py, chunker.py)
        │  → Chunk objects (text, source, location, doc_date)
        ▼
  ingest/classify.py
        │  → tags each chunk with an RBAC category + injection flag
        ▼
  rag/embed_store.py
        │  → local sentence-transformers embeddings → persisted in Chroma
        ▼
  scripts/build_index.py  (orchestrates the above + writes
        │                  data/understanding/*.summary.json)
        ▼
  ┌─────────────────────────────────────────────┐
  │  Query time (app.py → rag/answer.py)         │
  │                                               │
  │  memory/conversation.py:                      │
  │    prior (query, answer) turns for the        │
  │    active role are prepended to the prompt     │
  │    so follow-ups work  ◄── bucketed per role   │
  │    so switching roles can't leak a prior       │
  │    role's answer into a new role's context     │
  │                                               │
  │  rag/retriever.py:                            │
  │    1. vector search (top ~25 candidates)      │
  │    2. rag/rbac.py filters by role  ◄── RBAC   │
  │       enforced HERE, before the LLM ever      │
  │       sees the text                           │
  │    3. feedback/store.py reranks by past        │
  │       thumbs up/down on similar queries        │
  │    4. top_k chunks + history → prompt → Groq Llama 3 │
  │       (tools/currency.py bound via tool        │
  │       calling; model may request a             │
  │       convert_currency call before answering)  │
  └─────────────────────────────────────────────┘
        │
        ▼
  Answer + sources + tool calls + feedback buttons (Streamlit)
        │
        ▼
  feedback/store.py (SQLite) → boosts/few-shots future retrieval
  memory/conversation.py (in-session) → context for the next follow-up
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY
```

Only the LLM calls (answering + document summarization) need the Groq
API key. Embeddings run locally via `sentence-transformers`, so ingestion
works even without a key (summaries will just say "skipped").

## Run it

```bash
# 1. Generate synthetic sample data (Apple-style 10-Ks + quarterly Excel files)
python scripts/generate_sample_data.py

# 2. Build the index (chunk, classify, embed, summarize)
python scripts/build_index.py

# 3. Launch the app
streamlit run app.py
```

Then in the browser: pick a role in the sidebar (CEO / CTO / Analyst), ask a
question, look at the sources it cites, and use 👍/👎 to give feedback.

### Try the RBAC boundary
- As **CEO**, ask: *"What was our total compensation expense in FY2024?"* → answered.
- As **CTO**, ask the same question → refused, because `headcount_comp` isn't
  in the CTO's allowed categories (see `config.py`).
- As **Analyst**, ask about strategy or forward guidance → also refused.

### Try the feedback loop
1. Ask a question, hit 👎, and type a correction.
2. Ask the *same or a similarly worded* question again.
3. The correction is injected into the prompt as guidance (see
   `feedback/store.get_few_shot_examples`), and chunks tied to the
   negatively-rated answer are down-weighted in retrieval next time.

### Try conversation memory
1. Ask: *"What was iPhone revenue in Q3 2024?"*
2. Then ask a follow-up in the same role without repeating context: *"And
   what about Q4?"* or *"Convert that figure to EUR"* — the model has the
   prior turn available and resolves "that"/"Q3 → Q4" correctly.
3. Switch the role dropdown and ask a follow-up — memory for the new role
   starts empty; the previous role's turns are never replayed into it (see
   `memory/conversation.py`). Use "Clear conversation memory" in the
   sidebar to reset explicitly.

### Try tool calling (currency conversion)
Ask something like *"What was total revenue in FY2024, and what's that in
GBP?"* The model answers the revenue question from retrieved chunks as
normal, then calls the `convert_currency` tool (`tools/currency.py`, backed
by the free Frankfurter FX-rate API) to do the conversion rather than
guessing an exchange rate. Expand "🔧 Tools used" under the answer to see
the exact call and result. This requires outbound internet access to
`api.frankfurter.app`; if that's blocked, the tool returns an error message
that gets surfaced back to the model instead of crashing the app.

### Use real data instead of synthetic
Drop real 10-K PDFs and quarterly financial Excel files for any public
company into `data/raw/` (any filenames), then re-run `python
scripts/build_index.py`. Nothing else needs to change — ingestion doesn't
care about the company, only the file type.

## What's precomputed vs. computed on the fly

**Precomputed at ingestion** (`scripts/build_index.py`, run once per new document):
- Chunking, RBAC category classification, injection-pattern scanning
- Chunk embeddings (stored in Chroma)
- One LLM-generated summary per source document (`data/understanding/*.json`)
  — a cheap way to answer "what's in this document" without a full retrieval
  pass, and useful context for a human reviewing what got ingested.

**Computed per-query** (`rag/retriever.py`, `rag/answer.py`):
- The query embedding and vector search
- RBAC filtering (must be per-query, since it depends on who's asking)
- Feedback similarity search and score adjustment
- The final LLM call that generates the answer

Rationale: anything that doesn't depend on *who's asking* or *what's being
asked* is done once at ingestion, since it's the same result every time and
ingestion-time cost is amortized across many future queries. Anything that
depends on the specific question or the specific user's role has to happen
at query time.

## RBAC design

Enforcement lives in `rag/rbac.py` and is applied inside `rag/retriever.py`
**before** any chunk text is assembled into a prompt (see `rag/answer.py`).
This is a data-layer control, not a prompt instruction — a restricted chunk
is structurally excluded from the candidate set the LLM ever sees, so it
can't be leaked by asking the model to combine sources, roleplay a different
role, or get creative about phrasing. The role itself comes only from the
UI's `st.selectbox` (never parsed from the chat text), so a user can't
grant themselves more access by typing "pretend I'm the CEO."

Roles (see `config.py`, easy to extend):
| Role | Access |
|---|---|
| CEO | all categories |
| CTO | all except `headcount_comp` |
| Analyst | only `revenue`, `product`, `general` |

## Feedback loop design

Every answer can be rated 👍/👎 (with an optional free-text correction),
stored in SQLite (`feedback/store.py`). On future queries:
- Embeddings of past queries are compared (cosine similarity) to the new
  query to find "similar past questions."
- Chunks tied to a 👍 answer to a similar question get a retrieval score
  boost; chunks tied to a 👎 get a penalty.
- Up to 3 similar past examples (positive answers or corrections) are
  injected into the prompt as few-shot guidance.

This is deliberately simple and fully inspectable — every behavior change
traces back to a specific row in `feedback.sqlite3`. No fine-tuning, no
opaque state.

## Observability with LangSmith (optional)

Since answering and summarization run through LangChain's
`ChatGroq`, every call is trace-ready for
[LangSmith](https://smith.langchain.com/) with zero code changes - just env
vars:

```bash
# in .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...      # from LangSmith Settings -> API Keys
LANGCHAIN_PROJECT=financial-rag-agent
```

With those set, every `llm.invoke()` call (both query answering and
document summarization) is automatically traced. On top of that, this repo
wires two things a bit further:

- **Tags & metadata per call** (`rag/answer.py`): each trace is tagged with
  the acting role (e.g. `role:CTO`) and carries metadata for the query text,
  number of chunks used, and number of chunks blocked by RBAC - so in the
  LangSmith UI you can filter/search traces by role and spot-check that a
  CTO or Analyst trace's prompt never actually contains headcount/comp text.
- **Feedback attached to traces**: when a user hits 👍/👎 in the app, the
  corresponding LangSmith run_id (captured via `collect_runs()`) is stored
  in SQLite and also pushed to LangSmith as feedback
  (`feedback/store._push_feedback_to_langsmith`), so the human judgment
  lives right next to the trace it's judging - not just in a separate local
  table.

Leave `LANGCHAIN_TRACING_V2` unset (or `false`) and the app behaves exactly
the same, just without traces - this is fully optional and fails silently
if misconfigured.



Three layers, all in `ingest/classify.py` and `rag/injection_guard.py`:
1. **Ingestion-time scan**: chunks matching common injection patterns
   ("ignore previous instructions", "you are now", etc.) are flagged in
   metadata but still retrievable (so a legitimate chunk about, say, a
   company discussing security incidents isn't silently dropped).
2. **Prompt framing**: retrieved chunks are wrapped in explicit
   `<document_chunk>` tags with a system-prompt instruction that content in
   those tags is data, not commands — flagged chunks get an extra inline
   warning.
3. **Role authority**: the role is a trusted parameter from the app session,
   never derived from user or document text, so no injected or typed
   instruction can escalate access.

## What breaks at 100x scale

- **Chroma (local, single-file persistent client)** doesn't horizontally
  scale or handle concurrent writes well — would move to a managed vector
  DB (pgvector, Pinecone, Weaviate) with proper sharding.
- **SQLite feedback store** has the same problem — fine for a demo, needs
  Postgres + connection pooling for concurrent users.
- **Keyword-based classification** (`ingest/classify.py`) is fast and free
  but brittle against phrasing it doesn't expect, and a single chunk can
  legitimately span two categories (e.g., a paragraph mentioning both
  revenue and headcount) — at scale this needs either a small fine-tuned
  classifier or batched LLM calls with proper multi-label support, and
  probably a smaller chunk size to reduce category bleed.
- **Feedback similarity search is brute-force** (loads all rows, computes
  cosine sim in Python) — fine for hundreds of feedback rows, falls over at
  tens of thousands. Would move to an ANN index (e.g., store feedback
  embeddings in Chroma too) instead of a linear scan.
- **One LLM call per document for summaries** at ingestion time doesn't
  parallelize by default here — at scale, batch/async these calls.
- **No real auth** — role is a UI dropdown for this demo. Production needs
  real authentication (SSO/JWT) mapped to roles server-side, not
  client-selectable.
- **No eval harness** — RBAC correctness is currently verified by manual
  testing (see "Try the RBAC boundary" above). At scale you'd want an
  automated red-team suite that tries every role against every restricted
  category and asserts no leakage, run in CI on every ingestion pipeline
  change.

## What I'd do differently with more time

- Hybrid retrieval (BM25/keyword + vector) — pure embedding search misses
  exact figures like "$391,035 million" if phrasing differs slightly.
- Multi-label chunk classification instead of first-match-wins.
- A proper eval set of (role, question, expected-refusal-or-not) pairs run
  automatically against the RBAC layer.
- Streaming responses in the UI instead of a blocking spinner.
- Real authentication instead of a role dropdown.

## Repo structure

```
config.py                   # RBAC map, categories, model/paths config
ingest/
  chunker.py                 # word-based chunking with overlap
  classify.py                 # RBAC category tagging + injection scan
  pdf_ingest.py                # PDF → chunks (text + flattened tables)
  xlsx_ingest.py                 # Excel rows → sentence-like chunks
rag/
  embed_store.py              # local embeddings + Chroma persistence
  rbac.py                       # the actual access-control filter
  retriever.py                    # vector search → RBAC filter → feedback rerank
  injection_guard.py                # prompt framing defenses
  answer.py                           # prompt assembly + Groq call
feedback/
  store.py                    # SQLite feedback + similarity boosting/few-shot
scripts/
  generate_sample_data.py     # synthetic Apple-style PDFs/Excel for demo
  build_index.py                # ingestion pipeline orchestrator (run this to (re)index)
app.py                       # Streamlit UI
data/raw/                    # source PDFs/Excel (synthetic by default)
data/understanding/          # generated per-document summaries (JSON)
chroma_db/                   # generated vector index (gitignore this)
feedback.sqlite3             # generated feedback DB (gitignore this)
```
