"""Text splitting for clause-level streaming."""

import re

# Split on any natural pause: sentence endings, commas, semicolons, colons, dashes
_CLAUSE_RE = re.compile(r'(?<=[.!?,;:\u2014—-])\s+')


def split_clauses(text: str) -> list[str]:
    """Split text into clauses at any natural pause point for streaming."""
    parts = _CLAUSE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]
