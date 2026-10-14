"""Rough CPU processing estimates for the supported Whisper models.

These factors are planning hints for the dashboard. The actual runtime depends
on the machine, the recording, and whether the model weights are already cached.
"""

from __future__ import annotations

_REALTIME_FACTORS: dict[str, float] = {
    "tiny": 0.2,
    "base": 0.5,
    "small": 1.0,
}


def estimate_processing_seconds(duration_seconds: float | None, model: str) -> float | None:
    """Estimate how long transcription may take for a recording.

    Returns None when the duration is unknown.
    """
    if duration_seconds is None or duration_seconds < 0:
        return None
    factor = _REALTIME_FACTORS.get(model, 0.5)
    return round(duration_seconds * factor, 1)
