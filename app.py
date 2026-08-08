"""
Minimal Streamlit UI: pick a role, ask a question, see the answer + sources,
give thumbs up/down feedback (optionally with a correction).

Run: streamlit run app.py
"""

import streamlit as st

import config
from rag.answer import answer_question
from feedback.store import record_feedback, get_all_feedback_for_display
from memory.conversation import ConversationMemory

st.set_page_config(page_title="Financial Data Assistant", layout="wide")
st.title("📊 Financial Data Assistant")
st.caption("Agentic RAG over financial filings, with RBAC and a feedback loop.")

with st.sidebar:
    st.header("Session")
    role = st.selectbox("Acting as role", config.ROLES, index=0)
    st.markdown(f"**Allowed data categories:**")
    allowed = config.allowed_categories_for(role)
    st.write(", ".join(sorted(allowed)))
    st.divider()
    st.caption(
        "RBAC note: this dropdown is the only way role is set. Nothing typed "
        "in the chat box can change it - retrieval is filtered by role before "
        "any document text reaches the model."
    )

    st.divider()
    with st.expander("Recent feedback (last 50)"):
        for fb in get_all_feedback_for_display():
            icon = "👍" if fb["rating"] == "up" else "👎"
            st.write(f"{icon} [{fb['role']}] {fb['query']}")
            if fb["correction"]:
                st.caption(f"↳ correction: {fb['correction']}")

    st.divider()
    st.caption(
        "Conversation memory: the last few turns for the currently selected "
        "role are replayed into the prompt so you can ask follow-ups "
        "(e.g. \"convert that to EUR\"). Memory is kept separate per role, "
        "so switching roles never leaks a prior role's answer into a new "
        "role's context."
    )
    if st.button("Clear conversation memory"):
        st.session_state.memory.clear()
        st.success("Conversation memory cleared.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: query, role, answer, sources, chunk_ids

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

query = st.text_input("Ask a question about the financials", placeholder="e.g. What was iPhone revenue in Q1 2024?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and query.strip():
    with st.spinner("Retrieving data and generating answer..."):
        try:
            result = answer_question(query, role, memory=st.session_state.memory)
        except RuntimeError as e:
            st.error(str(e))
            result = None

    if result:
        st.session_state.history.insert(0, {
            "query": query,
            "role": role,
            "answer": result["answer"],
            "sources": result["sources"],
            "chunk_ids": result["chunks_used"],
            "num_blocked": result["num_blocked_by_rbac"],
            "run_id": result.get("run_id"),
            "tool_calls": result.get("tool_calls", []),
        })
        # Record this turn in per-role memory so follow-up questions in the
        # same role's context can refer back to it.
        st.session_state.memory.add_turn(role, query, result["answer"])

for i, item in enumerate(st.session_state.history):
    st.markdown("---")
    st.markdown(f"**Q ({item['role']}):** {item['query']}")
    st.markdown(item["answer"])

    if item["sources"]:
        with st.expander("Sources used"):
            for s in item["sources"]:
                st.write(f"- {s['source']} — {s['location']}")

    if item.get("tool_calls"):
        with st.expander("🔧 Tools used"):
            for tc in item["tool_calls"]:
                args_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                st.write(f"- `{tc['name']}({args_str})` → {tc['result']}")

    if item["num_blocked"]:
        st.caption(f"ℹ️ {item['num_blocked']} retrieved chunk(s) were withheld by RBAC for this role.")

    if item.get("run_id") and config.LANGSMITH_ENABLED:
        st.caption(f"🔍 LangSmith run: `{item['run_id']}` (search this ID in your LangSmith project)")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("👍", key=f"up_{i}"):
            record_feedback(item["query"], item["role"], item["answer"], "up",
                             item["chunk_ids"], run_id=item.get("run_id"))
            st.success("Thanks - noted as helpful.")
    with col2:
        if st.button("👎", key=f"down_{i}"):
            st.session_state[f"show_correction_{i}"] = True

    if st.session_state.get(f"show_correction_{i}"):
        correction = st.text_area("What should the answer have been?", key=f"correction_text_{i}")
        if st.button("Submit correction", key=f"submit_correction_{i}"):
            record_feedback(item["query"], item["role"], item["answer"], "down",
                             item["chunk_ids"], correction=correction, run_id=item.get("run_id"))
            st.session_state[f"show_correction_{i}"] = False
            st.success("Thanks - this will inform future similar answers.")
