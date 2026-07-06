"""Text splitting for clause-level streaming."""

import re

# Split on any natural pause: sentence endings, commas, semicolons, colons, dashes
_CLAUSE_RE = re.compile(r'(?<=[.!?,;:\u2014\u2013—-])\s+')

# Orphan: a clause that is just a number+punct, e.g. "1." or "2:"
_ORPHAN_RE = re.compile(r'^\d+[.:]$')

# Trailing separator chars that Kokoro should not voice (keep .!? for prosody)
_TRAILING_SEP_RE = re.compile('[,;:\u2014\u2013-]+$')

_MIN_CHARS = 10


def _preprocess(text: str) -> str:
    """Normalize structure before regex splitting."""
    # Numbered list items: "\n1. foo\n2. bar" → ". foo. bar"
    # Each list-item newline+prefix becomes a sentence boundary
    text = re.sub(r'\n\d+\.\s+', '. ', text)
    # Paragraph breaks → sentence boundary
    text = re.sub(r'\r?\n\s*\r?\n+', '. ', text)
    # Remaining mid-line newlines → space
    text = re.sub(r'\r?\n', ' ', text)
    # Collapse punctuation artifacts from the substitutions above:
    # "cause.. " / "danger: . " / "that?. " → single punct + space
    text = re.sub(r'([.!?:,;])\s*\.\s+', r'\1 ', text)
    return text


def _postprocess(parts: list[str]) -> list[str]:
    """Merge fragments that shouldn't stand alone."""
    result: list[str] = []
    for clause in parts:
        if not result:
            result.append(clause)
            continue
        words = clause.split()
        prev = result[-1]
        should_merge = (
            len(clause) < _MIN_CHARS
            or len(words) <= 1
            or _ORPHAN_RE.match(clause)
        )
        if should_merge:
            result[-1] = prev.rstrip() + ' ' + clause.lstrip()
        else:
            result.append(clause)
    return result


def split_clauses(text: str) -> list[str]:
    """Split text into clauses at any natural pause point for streaming."""
    text = _preprocess(text.strip())
    parts = _CLAUSE_RE.split(text)
    parts = [s.strip() for s in parts if s.strip()]
    parts = _postprocess(parts)
    return [_TRAILING_SEP_RE.sub('', p).strip() for p in parts]
