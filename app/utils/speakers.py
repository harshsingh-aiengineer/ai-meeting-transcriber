"""Pause-based speaker turn estimates.

These labels are not voice identification. A long silence starts the next
estimated speaker, cycling through the requested speaker count.
"""

from app.models.schemas import TranscriptSegment


def assign_speaker_turns(
    segments: list[TranscriptSegment],
    speaker_count: int,
    gap_seconds: float,
) -> list[TranscriptSegment]:
    """Return a copy of ``segments`` with estimated speaker labels.

    Raises:
        ValueError: The speaker count or gap is outside the supported range.
    """
    if speaker_count < 2 or speaker_count > 8:
        raise ValueError("Estimated speaker count must be between 2 and 8.")
    if gap_seconds <= 0 or gap_seconds > 10:
        raise ValueError("Speaker gap must be between 0 and 10 seconds.")

    speaker_index = 0
    previous_end: float | None = None
    labeled: list[TranscriptSegment] = []
    for segment in segments:
        if previous_end is not None and segment.start - previous_end >= gap_seconds:
            speaker_index = (speaker_index + 1) % speaker_count
        labeled.append(
            segment.model_copy(update={"speaker": f"Speaker {speaker_index + 1}"})
        )
        previous_end = segment.end
    return labeled
