"""Reference-transcript scoring for repeatable accuracy checks.

These helpers compare a hypothesis with a manual reference. They do not
transcribe audio and they do not claim a result for a recording that was
not scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import TranscriptSegment

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class AlignmentScores:
    """Word and character error counts against one reference."""

    wer: float
    cer: float
    deletions: int
    insertions: int
    substitutions: int
    reference_words: int
    hypothesis_words: int

    @property
    def omission_rate(self) -> float:
        """Share of reference words missing from the hypothesis."""
        if self.reference_words == 0:
            return 0.0
        return self.deletions / self.reference_words

    @property
    def insertion_rate(self) -> float:
        """Share of extra hypothesis words relative to the reference length."""
        if self.reference_words == 0:
            return 0.0
        return self.insertions / self.reference_words


def word_tokens(text: str) -> list[str]:
    """Return lowercase word tokens."""
    return [match.group(0).casefold() for match in _WORD.finditer(text)]


def score_transcripts(reference: str, hypothesis: str) -> AlignmentScores:
    """Score word and character error against a manual reference.

    Raises:
        ValueError: The reference text has no words.
    """
    reference_words = word_tokens(reference)
    hypothesis_words = word_tokens(hypothesis)
    if not reference_words:
        raise ValueError("The reference transcript has no words to score.")
    deletions, insertions, substitutions = _operation_counts(
        reference_words, hypothesis_words
    )
    word_distance = deletions + insertions + substitutions
    reference_chars = list(" ".join(reference_words))
    hypothesis_chars = list(" ".join(hypothesis_words))
    char_distance = sum(_operation_counts(reference_chars, hypothesis_chars))
    return AlignmentScores(
        wer=word_distance / len(reference_words),
        cer=char_distance / len(reference_chars),
        deletions=deletions,
        insertions=insertions,
        substitutions=substitutions,
        reference_words=len(reference_words),
        hypothesis_words=len(hypothesis_words),
    )


def speaker_overlap_accuracy(
    reference: list[TranscriptSegment],
    hypothesis: list[TranscriptSegment],
) -> float | None:
    """Return the time-weighted share of reference speech with a matching label.

    Returns None when either side is missing a speaker label. A missing label
    is not treated as a match or a mismatch.
    """
    if not reference or not hypothesis:
        return None
    if any(segment.speaker is None for segment in reference + hypothesis):
        return None
    total = 0.0
    matched = 0.0
    for segment in reference:
        duration = max(0.0, segment.end - segment.start)
        if duration == 0:
            continue
        total += duration
        label = _majority_label(segment, hypothesis)
        if label == segment.speaker:
            matched += duration
    if total == 0:
        return None
    return matched / total


def _majority_label(
    reference: TranscriptSegment,
    hypothesis: list[TranscriptSegment],
) -> str | None:
    overlap_by_speaker: dict[str, float] = {}
    for segment in hypothesis:
        overlap = _overlap(reference.start, reference.end, segment.start, segment.end)
        if overlap <= 0 or segment.speaker is None:
            continue
        overlap_by_speaker[segment.speaker] = (
            overlap_by_speaker.get(segment.speaker, 0.0) + overlap
        )
    if not overlap_by_speaker:
        return None
    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _operation_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        distances[row][0] = row
    for column in range(1, columns):
        distances[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = 0 if reference[row - 1] == hypothesis[column - 1] else 1
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1] + substitution_cost,
            )
    deletions = insertions = substitutions = 0
    row, column = len(reference), len(hypothesis)
    while row > 0 or column > 0:
        if (
            row > 0
            and column > 0
            and distances[row][column] == distances[row - 1][column - 1]
            and reference[row - 1] == hypothesis[column - 1]
        ):
            row -= 1
            column -= 1
        elif (
            row > 0
            and column > 0
            and distances[row][column] == distances[row - 1][column - 1] + 1
        ):
            substitutions += 1
            row -= 1
            column -= 1
        elif row > 0 and distances[row][column] == distances[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return deletions, insertions, substitutions
