"""Display-only cleanup for a transcript.

The functions here never add words. They collapse obvious recognition loops
and extra spaces. The stored transcript text stays as the model produced it.
"""

from __future__ import annotations

from app.models.schemas import Transcript


def clean_transcript_text(text: str) -> str:
    """Collapse repeated loops and extra spaces without adding words."""
    collapsed = _collapse_phrase_loops(_collapse_token_loops(text))
    return " ".join(collapsed.split())


def cleaned_transcript_text(transcript: Transcript) -> str:
    """Return a cleaned reading view built from the raw segments."""
    parts = [
        clean_transcript_text(segment.text)
        for segment in transcript.segments
        if segment.text.strip()
    ]
    if parts:
        return " ".join(part for part in parts if part).strip()
    return clean_transcript_text(transcript.text)


def _collapse_token_loops(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        word = tokens[index]
        run = 1
        while (
            index + run < len(tokens)
            and tokens[index + run].casefold() == word.casefold()
        ):
            run += 1
        if run >= 3:
            kept.append(word)
            index += run
        else:
            kept.append(word)
            index += 1
    return " ".join(kept)


def _collapse_phrase_loops(text: str) -> str:
    tokens = text.split()
    changed = True
    while changed and tokens:
        changed = False
        index = 0
        kept: list[str] = []
        while index < len(tokens):
            removed = False
            for size in range(min(8, (len(tokens) - index) // 2), 2, -1):
                left = [token.casefold() for token in tokens[index : index + size]]
                right = [
                    token.casefold()
                    for token in tokens[index + size : index + (2 * size)]
                ]
                if left == right:
                    kept.extend(tokens[index : index + size])
                    index += 2 * size
                    removed = True
                    changed = True
                    break
            if not removed:
                kept.append(tokens[index])
                index += 1
        tokens = kept
    return " ".join(tokens)
