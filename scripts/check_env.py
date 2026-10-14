"""Check the local runtime without starting the web app."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    """Print Python, ffmpeg, and optional Ollama status."""
    print(f"Python {sys.version.split()[0]}")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    print(f"ffmpeg: {ffmpeg or 'not found'}")
    print(f"ffprobe: {ffprobe or 'not found'}")

    from app.config import get_settings
    from app.services.llm import OllamaClient

    settings = get_settings()
    print(f"Whisper model: {settings.whisper_model}")
    print(f"Device: {settings.device}")
    print(f"Compute type: {settings.compute_type}")
    client = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=3,
    )
    print(f"Ollama: {client.health().message}")

    if ffmpeg is None or ffprobe is None:
        print("Install ffmpeg before transcribing recordings.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
