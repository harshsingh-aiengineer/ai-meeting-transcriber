"""Tests for transcription option checks without loading a model."""

from pathlib import Path

import pytest

from app.services.transcription import TranscriptionService, transcribe_file
from app.utils.uploads import ALLOWED_EXTENSIONS, redact_user_message


def test_invalid_language_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Language"):
        transcribe_file(
            tmp_path / "missing.wav",
            meeting_id="c" * 32,
            source_name="missing.wav",
            model_name="base",
            device="cpu",
            compute_type="int8",
            language="english",
            estimate_speakers=False,
            speaker_count=2,
            gap_seconds=1.2,
        )


def test_format_segments_uses_start_end_text() -> None:
    from app.models.schemas import TranscriptSegment

    service = TranscriptionService("base", "cpu", "int8")
    formatted = service.format_segments(
        [TranscriptSegment(start=0, end=4.8, text="Hello team")]
    )
    assert formatted == [{"start": 0.0, "end": 4.8, "text": "Hello team"}]
    with_speaker = service.format_segments(
        [TranscriptSegment(start=1.23456, end=2, text="Noted", speaker="Speaker 1")]
    )
    assert with_speaker == [
        {"start": 1.235, "end": 2.0, "text": "Noted", "speaker": "Speaker 1"}
    ]


def test_save_transcript_omits_filesystem_paths(tmp_path: Path) -> None:
    import json

    from app.models.schemas import Transcript, TranscriptSegment

    service = TranscriptionService("base", "cpu", "int8")
    transcript = Transcript(
        meeting_id="d" * 32,
        source_name="notes.mp3",
        model="base",
        device="cpu",
        compute_type="int8",
        segments=[TranscriptSegment(start=0, end=1.5, text="Hello")],
        text="Hello",
    )
    path = service.save_transcript(transcript, tmp_path / "transcript.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["segments"][0]["text"] == "Hello"
    assert "C:\\" not in path.read_text(encoding="utf-8")
    assert "source_path" not in payload


def test_load_model_uses_the_process_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str]] = []
    sentinel = object()

    def fake_get_model(model_name: str, device: str, compute_type: str) -> object:
        calls.append((model_name, device, compute_type))
        return sentinel

    monkeypatch.setattr("app.services.transcription.get_model", fake_get_model)
    service = TranscriptionService("base", "cpu", "int8")
    assert service.load_model() is sentinel
    assert service.load_model() is sentinel
    assert calls == [("base", "cpu", "int8")]


def test_user_messages_hide_local_paths() -> None:
    message = redact_user_message(r"ffmpeg failed: C:\Users\harsh\recording.mp4")
    assert r"C:\Users" not in message
    assert "a local file" in message


def test_common_meeting_formats_are_allowed() -> None:
    for extension in (".mp3", ".wav", ".m4a", ".mp4", ".webm"):
        assert extension in ALLOWED_EXTENSIONS
