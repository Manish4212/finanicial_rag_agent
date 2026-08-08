"""
feedback: the learning-from-adoption loop.

Modules:
    store - SQLite-backed storage for thumbs up/down + corrections, plus
            similarity-based retrieval boosting and few-shot example lookup.

Import from the submodule directly, e.g.:
    from feedback.store import record_feedback, get_feedback_boosts
"""

from feedback.store import (
    record_feedback,
    get_feedback_boosts,
    get_few_shot_examples,
    get_all_feedback_for_display,
)

__all__ = [
    "record_feedback",
    "get_feedback_boosts",
    "get_few_shot_examples",
    "get_all_feedback_for_display",
]
