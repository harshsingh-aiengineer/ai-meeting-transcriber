"""ffmpeg helpers for converting meeting recordings to WAV audio."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot process a recording."""


def require_tool(name: str) -> str:
    """Return the executable path for a required media tool."""
    located = shutil.which(name)
    if located is None:
        raise MediaError(
            f"{name} is not installed or not on PATH. "
            "Install ffmpeg (it includes ffprobe) and restart the app."
        )
    return located


def build_extract_command(ffmpeg: str, source: Path, destination: Path) -> list[str]:
    """Build the ffmpeg command that writes 16 kHz mono WAV."""
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(destination),
    ]


def build_duration_command(ffprobe: str, source: Path) -> list[str]:
    """Build the ffprobe command that prints duration in seconds."""
    return [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]


def run_command(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a media command and raise ``MediaError`` on failure."""
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{Path(command[0]).name} timed out.") from exc
    except FileNotFoundError as exc:
        raise MediaError("ffmpeg or ffprobe could not be started.") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown error").strip()
        raise MediaError(f"{Path(command[0]).name} failed: {detail[-800:]}")
    return completed


def probe_duration(source: Path) -> float | None:
    """Return the media duration in seconds, or None when it cannot be read."""
    ffprobe = require_tool("ffprobe")
    completed = run_command(build_duration_command(ffprobe, source), timeout=60)
    raw = completed.stdout.strip()
    try:
        duration = float(raw)
    except ValueError:
        return None
    if duration < 0:
        return None
    return duration


def extract_audio(source: Path, destination: Path) -> None:
    """Convert any supported recording to a 16 kHz mono WAV file."""
    ffmpeg = require_tool("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_extract_command(ffmpeg, source, destination), timeout=1800)
