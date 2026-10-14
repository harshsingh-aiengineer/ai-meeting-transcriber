"""Tests for ffmpeg command construction and audio extraction."""

import shutil
import wave
from pathlib import Path

import pytest

from app.utils.media import MediaError, build_extract_command, extract_audio, require_tool


def test_extract_command_uses_mono_16k_wav() -> None:
    command = build_extract_command("ffmpeg", Path("in.mp4"), Path("out.wav"))
    assert command[:3] == ["ffmpeg", "-y", "-hide_banner"]
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-f") + 1] == "wav"


def test_require_tool_reports_a_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(MediaError, match="ffmpeg"):
        require_tool("ffmpeg")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_extract_audio_from_wav(tmp_path: Path) -> None:
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    destination = tmp_path / "converted.wav"
    extract_audio(source, destination)
    assert destination.is_file()
    assert destination.stat().st_size > 44
