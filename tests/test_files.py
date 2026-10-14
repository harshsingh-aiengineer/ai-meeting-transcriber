"""Tests for pause-based speaker labels and upload checks."""

from pathlib import Path

import pytest

from app.models.schemas import TranscriptSegment
from app.utils.speakers import assign_speaker_turns
from app.utils.uploads import resolve_under_directory, sanitize_filename, save_upload


def test_speaker_turns_switch_after_a_pause() -> None:
    segments = [
        TranscriptSegment(start=0, end=1, text="Hello"),
        TranscriptSegment(start=1.2, end=2, text="Still here"),
        TranscriptSegment(start=4, end=5, text="A new turn"),
    ]
    labeled = assign_speaker_turns(segments, speaker_count=2, gap_seconds=1.2)
    assert [segment.speaker for segment in labeled] == [
        "Speaker 1",
        "Speaker 1",
        "Speaker 2",
    ]
    assert segments[0].speaker is None


def test_sanitize_filename_removes_path_pieces() -> None:
    assert sanitize_filename(r"..\Q3 Review.mp4") == "Q3_Review.mp4"


def test_save_upload_rejects_empty_oversize_and_unknown_type(tmp_path: Path) -> None:
    meeting_id = "a" * 32
    with pytest.raises(ValueError, match="empty"):
        save_upload(b"", "notes.wav", tmp_path, meeting_id, max_bytes=100)
    with pytest.raises(ValueError, match="above"):
        save_upload(b"12345", "notes.wav", tmp_path, meeting_id, max_bytes=4)
    with pytest.raises(ValueError, match="Unsupported"):
        save_upload(b"12345", "notes.exe", tmp_path, meeting_id, max_bytes=100)
    with pytest.raises(ValueError, match="Unsupported"):
        save_upload(b"12345", r"..\notes.exe", tmp_path, meeting_id, max_bytes=100)


def test_save_upload_writes_a_safe_name(tmp_path: Path) -> None:
    meeting_id = "b" * 32
    path = save_upload(b"abc", "My Meeting.wav", tmp_path, meeting_id, max_bytes=100)
    assert path.parent == tmp_path
    assert path.name == f"{meeting_id}_My_Meeting.wav"
    assert path.read_bytes() == b"abc"


def test_resolve_under_directory_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        resolve_under_directory(root, outside)
