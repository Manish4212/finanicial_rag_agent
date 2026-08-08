"""
Short-term conversational memory: lets the assistant answer follow-up
questions ("what about Q4 instead?", "convert that to EUR") without the
user having to restate context.

Design notes / why this isn't a single global history list:

RBAC is enforced per-query in rag/retriever.py based on the role active at
the time of that query. If a CEO asks about headcount/comp and then the
role dropdown is switched to Analyst, a naive shared history would replay
the CEO's answer text back into the prompt for the Analyst turn - and the
model could repeat or build on data that role should never see, even
though retrieval for the *new* query would correctly withhold it. That's a
leak through conversation history rather than retrieval.

So memory is bucketed by role. Switching roles effectively starts a new
conversation for RBAC purposes; each role keeps its own turn history.

This is intentionally a lightweight, in-process store (a plain dict held
in Streamlit's session_state, not sqlite) - it only needs to live for the
duration of one browser session, unlike feedback which is durable across
sessions/users. See feedback/store.py for the persistent counterpart.
"""

from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import config


class ConversationMemory:
    """Holds recent (query, answer) turns, bucketed by role."""

    def __init__(self, max_turns: int = None):
        self.max_turns = max_turns or config.MEMORY_MAX_TURNS
        self._turns_by_role: Dict[str, List[dict]] = {}

    def add_turn(self, role: str, query: str, answer: str) -> None:
        turns = self._turns_by_role.setdefault(role, [])
        turns.append({"query": query, "answer": answer})
        # Keep only the most recent max_turns turns for this role.
        if len(turns) > self.max_turns:
            del turns[: len(turns) - self.max_turns]

    def get_turns(self, role: str) -> List[dict]:
        return list(self._turns_by_role.get(role, []))

    def clear(self, role: str = None) -> None:
        """Clear memory for one role, or everything if role is None."""
        if role is None:
            self._turns_by_role.clear()
        else:
            self._turns_by_role.pop(role, None)

    def as_messages(self, role: str) -> List[BaseMessage]:
        """Turn this role's stored history into alternating Human/AI messages,
        ready to splice into the message list before the current question."""
        messages: List[BaseMessage] = []
        for turn in self.get_turns(role):
            messages.append(HumanMessage(content=turn["query"]))
            messages.append(AIMessage(content=turn["answer"]))
        return messages
