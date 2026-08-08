"""
Classifies each chunk into one of config.CATEGORIES so RBAC can filter on it,
and flags chunks that look like they contain prompt-injection attempts.

Design choice: this uses cheap keyword heuristics instead of an LLM call per
chunk. At real scale you'd swap this for a small fine-tuned classifier or a
batched LLM call - the interface (classify_chunk -> category string) stays
the same either way, so it's a drop-in swap. Keeping it keyword-based here
means ingestion has zero API cost and zero API-key dependency.
"""

import re

# Ordered: first matching category wins. Order matters - more specific first.
CATEGORY_KEYWORDS = {
    "headcount_comp": [
        "headcount", "employee count", "full-time employees", "compensation",
        "salary", "salaries", "wages", "bonus", "stock-based compensation",
        "benefits expense", "payroll", "workforce size",
    ],
    "guidance": [
        "guidance", "outlook", "we expect", "we anticipate", "forward-looking",
        "next quarter", "next fiscal year", "projected",
    ],
    "strategy": [
        "strategic", "acquisition", "merger", "competitive landscape",
        "market position", "long-term strategy", "roadmap", "partnership",
    ],
    "revenue": [
        "revenue", "net sales", "gross margin", "operating income",
        "segment results", "earnings per share", "cost of sales",
    ],
    "product": [
        "iphone", "ipad", "mac", "wearables", "services segment", "unit sales",
        "product line", "product category",
    ],
}

# Patterns that suggest an ingested document is trying to inject instructions
# into the model rather than just describing financial data.
INJECTION_PATTERNS = [
    r"ignore (all|any|previous|the above) instructions",
    r"disregard (all|any|previous|the above)",
    r"you (are|must) now",
    r"system\s*:",
    r"assistant\s*:",
    r"reveal (the )?(system prompt|instructions)",
    r"act as (if|a)",
    r"do not (follow|apply) (the )?(access|rbac|permission)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def classify_chunk(text: str) -> str:
    lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return category
    return "general"


def scan_for_injection(text: str) -> bool:
    """Return True if text contains a likely prompt-injection attempt."""
    return bool(_INJECTION_RE.search(text))
