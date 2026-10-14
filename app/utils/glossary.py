"""Turn a user-supplied name list into a short Whisper hint string."""

from __future__ import annotations

import re

_MAX_TERMS = 24
_MAX_TERM_CHARS = 48
_MAX_HINT_CHARS = 400


def format_glossary(text: str | None) -> str:
    """Return a comma-separated hint list.

    Empty input returns an empty string. Duplicate and very long terms are
    dropped so the hint cannot crowd out the audio.
    """
    if not text:
        return ""
    terms: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\n,;]+", text):
        term = " ".join(part.split())
        if not term or len(term) > _MAX_TERM_CHARS:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= _MAX_TERMS:
            break
    return ", ".join(terms)[:_MAX_HINT_CHARS].rstrip(", ")
