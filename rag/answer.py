"""Builds the final prompt from retrieved+RBAC-filtered chunks and calls Groq via LangChain."""

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tracers.context import collect_runs

import config
from rag.retriever import retrieve
from rag.injection_guard import INJECTION_SYSTEM_NOTE, wrap_chunk_for_prompt
from feedback.store import get_few_shot_examples
from memory.conversation import ConversationMemory
from tools.currency import CURRENCY_TOOLS

_llm = None
_llm_with_tools = None

# Safety cap on the tool-call round trips we allow per question, so a
# misbehaving/looping model can't spin forever burning API calls.
MAX_TOOL_ROUNDS = 3


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file or environment."
            )
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=config.GROQ_API_KEY,
            temperature=0,
            max_tokens=1000,
        )
    return _llm


def get_llm_with_tools() -> ChatGroq:
    """Same chat model, with the tool-calling tools (currently: currency
    conversion) bound so the model can request calls to them."""
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(CURRENCY_TOOLS, parallel_tool_calls=False)
    return _llm_with_tools


# name -> callable tool, for dispatching tool_calls the model asks for.
_TOOLS_BY_NAME = {t.name: t for t in CURRENCY_TOOLS}


SYSTEM_PROMPT_TEMPLATE = """You are a financial data assistant answering questions for a user with the \
role "{role}".

{injection_note}

Answer ONLY using the retrieved document_chunk data provided below (and, for \
earlier turns in this conversation, your own prior answers). If the \
retrieved chunks do not contain enough information to answer - including if \
the answer would require data that was withheld because of this user's role \
- say clearly that the information is not available to you, and do not \
guess, estimate, or fill the gap with general knowledge. Never say why data \
was withheld or hint that restricted data exists; simply say the requested \
information isn't available.

When you do answer, cite the source and location of the chunk(s) you used, \
e.g. "(Source: apple_10k_2024.pdf, page 12)".

You have access to a convert_currency tool. Use it whenever the user asks \
for a figure to be expressed in a different currency than it was reported \
in - never convert currency by estimating the rate yourself. Only use it on \
amounts that came from the retrieved document data or from earlier in this \
conversation; never invent a figure to convert.

{few_shot_block}"""


def build_few_shot_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["Here are examples of past questions and answers this system got positive feedback on "
             "(or corrections a user made) - use them as style/content guidance where relevant:"]
    for ex in examples:
        if ex["type"] == "positive":
            lines.append(f'- Q: "{ex["query"]}" -> A: "{ex["answer"]}" (rated helpful)')
        else:
            lines.append(
                f'- Q: "{ex["query"]}" -> a previous answer was corrected. '
                f'Correct answer/guidance: "{ex["correction"]}"'
            )
    return "\n".join(lines)


def answer_question(query_text: str, role: str, memory: ConversationMemory = None) -> dict:
    retrieval = retrieve(query_text, role)
    chunks = retrieval["chunks"]

    if not chunks:
        return {
            "answer": "I don't have information available to answer that for your role.",
            "sources": [],
            "num_blocked_by_rbac": retrieval["num_blocked_by_rbac"],
            "chunks_used": [],
            "run_id": None,
        }

    context_block = "\n\n".join(wrap_chunk_for_prompt(c, i) for i, c in enumerate(chunks))
    few_shot_examples = get_few_shot_examples(query_text, role)
    few_shot_block = build_few_shot_block(few_shot_examples)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        role=role,
        injection_note=INJECTION_SYSTEM_NOTE,
        few_shot_block=few_shot_block,
    )

    user_message = f"{context_block}\n\nQuestion: {query_text}"

    # Prior turns for this role (if any) go between the system prompt and
    # the current question, so follow-ups like "convert that to EUR" or
    # "what about the prior quarter" resolve correctly. See
    # memory/conversation.py for why this is bucketed per-role.
    history_messages = memory.as_messages(role) if memory is not None else []

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *history_messages,
        HumanMessage(content=user_message),
    ]

    llm_with_tools = get_llm_with_tools()

    invoke_config = {
        "tags": ["financial-rag-agent", f"role:{role}"],
        "metadata": {
            "role": role,
            "query": query_text,
            "num_chunks_used": len(chunks),
            "num_blocked_by_rbac": retrieval["num_blocked_by_rbac"],
        },
        "run_name": "answer_question",
    }

    tool_calls_made = []
    run_id = None

    # collect_runs() captures the LangSmith run_id for each call (a no-op
    # locally if LangSmith tracing isn't enabled). We attach role and query
    # as metadata/tags so traces are filterable in the LangSmith UI by role
    # - handy for auditing "did any CTO/Analyst trace ever end up with
    # headcount_comp text in its prompt" during a walkthrough.
    with collect_runs() as run_collector:
        response = llm_with_tools.invoke(messages, config=invoke_config)

        # Tool-calling loop: the model can ask for convert_currency (or
        # future tools) instead of answering directly. We execute the
        # requested tool(s) locally, feed the result back as a ToolMessage,
        # and let the model produce a final answer grounded in that result.
        # Capped at MAX_TOOL_ROUNDS so a looping model can't run away.
        rounds = 0
        while response.tool_calls and rounds < MAX_TOOL_ROUNDS:
            messages.append(response)
            for call in response.tool_calls:
                tool_fn = _TOOLS_BY_NAME.get(call["name"])
                if tool_fn is None:
                    result = f"Error: unknown tool '{call['name']}'."
                else:
                    try:
                        result = tool_fn.invoke(call["args"])
                    except Exception as e:  # noqa: BLE001 - surface any tool error to the model, not a crash
                        result = f"Error running tool '{call['name']}': {e}"
                tool_calls_made.append({"name": call["name"], "args": call["args"], "result": result})
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            response = llm_with_tools.invoke(messages, config=invoke_config)
            rounds += 1

        if run_collector.traced_runs:
            run_id = str(run_collector.traced_runs[-1].id)

    answer_text = response.content or ""

    return {
        "answer": answer_text,
        "sources": [{"source": c["source"], "location": c["location"]} for c in chunks],
        "num_blocked_by_rbac": retrieval["num_blocked_by_rbac"],
        "chunks_used": [f'{c["source"]}::{c["location"]}' for c in chunks],
        "run_id": run_id,
        "tool_calls": tool_calls_made,
    }