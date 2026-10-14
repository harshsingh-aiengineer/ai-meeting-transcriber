"""faster-whisper transcription.

The Whisper model is loaded on first use and cached for the process.
Importing this module does not download model weights.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import COMPUTE_TYPES, DEVICES
from app.models.schemas import Transcript, TranscriptSegment
from app.utils.glossary import format_glossary
from app.utils.media import MediaError, extract_audio, probe_duration
from app.utils.speakers import assign_speaker_turns

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
_MODEL_NAME = re.compile(r"^[A-Za-z0-9._:-]+(?:/[A-Za-z0-9._:-]+)?$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")
_MODELS: dict[tuple[str, str, str], Any] = {}
_LOCK = threading.Lock()
# Quieter speech stays in, and a short pause can start a new window.
# min_speech_duration_ms stays 0 so a brief reply is not discarded.
SPEECH_VAD: dict[str, float | int] = {
    "threshold": 0.35,
    "min_speech_duration_ms": 0,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 600,
}


class TranscriptionService:
    """Transcribe meeting recordings with a cached faster-whisper model.

    ``load_model`` keeps one model per process for a model, device, and compute
    type. Creating another service with the same settings reuses that model.
    """

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None

    def load_model(self) -> Any:
        """Return the cached Whisper model, loading it on first use.

        Raises:
            RuntimeError: The model cannot be downloaded or initialized.
        """
        if self._model is not None:
            return self._model
        try:
            self._model = get_model(self.model_name, self.device, self.compute_type)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Whisper model '{self.model_name}' "
                f"on {self.device} ({self.compute_type}). {exc}"
            ) from exc
        return self._model

    def transcribe(
        self,
        source_path: Path,
        *,
        meeting_id: str,
        source_name: str,
        language: str | None,
        estimate_speakers: bool,
        speaker_count: int,
        gap_seconds: float,
        glossary: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Transcript:
        """Transcribe a recording to timestamped segments.

        Raises:
            MediaError: ffmpeg cannot read or convert the file.
            RuntimeError: The Whisper model cannot be loaded or run.
            ValueError: A transcription option is invalid.
        """
        _validate_options(
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=language,
            estimate_speakers=estimate_speakers,
            speaker_count=speaker_count,
            gap_seconds=gap_seconds,
        )
        _emit(on_progress, 0.08, "Reading media duration.")
        try:
            probed = probe_duration(source_path)
        except MediaError as exc:
            logger.warning("Could not read media duration: %s", exc)
            probed = None

        with tempfile.TemporaryDirectory(prefix="meeting-audio-") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            _emit(on_progress, 0.15, "Extracting 16 kHz mono audio.")
            extract_audio(source_path, wav_path)
            _emit(
                on_progress,
                0.28,
                f"Loading Whisper model '{self.model_name}'. The first run downloads model files.",
            )
            model = self.load_model()
            _emit(on_progress, 0.4, "Transcribing speech.")
            segments, info = _transcribe_with_fallback(
                model,
                wav_path,
                language=language,
                glossary=glossary,
                on_progress=on_progress,
            )

        if estimate_speakers and segments:
            segments = assign_speaker_turns(segments, speaker_count, gap_seconds)
            speaker_mode = "pause_estimate"
        else:
            speaker_mode = "none"

        duration = probed if probed and probed > 0 else _optional_float(getattr(info, "duration", None))
        language_code = getattr(info, "language", None)
        probability = _optional_float(getattr(info, "language_probability", None))
        text = " ".join(segment.text for segment in segments).strip()
        _emit(on_progress, 0.96, "Transcript ready.")
        logger.info(
            "Transcription completed meeting_id=%s segments=%s language=%s duration=%s",
            meeting_id,
            len(segments),
            language_code,
            duration,
        )
        return Transcript(
            meeting_id=meeting_id,
            source_name=source_name,
            language=language_code if isinstance(language_code, str) else None,
            language_probability=probability,
            duration_seconds=duration,
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            speaker_mode=speaker_mode,
            segments=segments,
            text=text,
        )

    def format_segments(self, segments: list[TranscriptSegment]) -> list[dict[str, float | str]]:
        """Return timestamped segment objects for JSON export."""
        formatted: list[dict[str, float | str]] = []
        for segment in segments:
            item: dict[str, float | str] = {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text,
            }
            if segment.speaker:
                item["speaker"] = segment.speaker
            formatted.append(item)
        return formatted

    def save_transcript(self, transcript: Transcript, destination: Path) -> Path:
        """Write structured transcript JSON.

        The file contains segment times and text. It does not store absolute
        filesystem paths.

        Raises:
            ValueError: The destination file name is not safe.
        """
        if not re.fullmatch(r"[A-Za-z0-9._-]+", destination.name):
            raise ValueError("Transcript file name is invalid.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "language": transcript.language,
            "duration_seconds": transcript.duration_seconds,
            "model": transcript.model,
            "text": transcript.text,
            "segments": self.format_segments(transcript.segments),
        }
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination


def transcribe_file(
    source_path: Path,
    *,
    meeting_id: str,
    source_name: str,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    estimate_speakers: bool,
    speaker_count: int,
    gap_seconds: float,
    glossary: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> Transcript:
    """Transcribe a recording with a cached :class:`TranscriptionService`."""
    service = TranscriptionService(model_name, device, compute_type)
    return service.transcribe(
        source_path,
        meeting_id=meeting_id,
        source_name=source_name,
        language=language,
        estimate_speakers=estimate_speakers,
        speaker_count=speaker_count,
        gap_seconds=gap_seconds,
        glossary=glossary,
        on_progress=on_progress,
    )


def get_model(model_name: str, device: str, compute_type: str) -> Any:
    """Return a cached WhisperModel, loading it on first use."""
    key = (model_name, device, compute_type)
    with _LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            model_name,
            device,
            compute_type,
        )
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _MODELS[key] = model
        return model


def decode_options(glossary: str | None) -> dict[str, Any]:
    """Return faster-whisper settings for one recording.

    Previous-text prompting is off because it is the documented cause of
    repetition loops. A glossary, when supplied, is passed as a hint and does
    not replace the audio.
    """
    options: dict[str, Any] = {
        "beam_size": 5,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "vad_parameters": dict(SPEECH_VAD),
        "language_detection_segments": 3,
    }
    hint = format_glossary(glossary)
    if hint:
        options["hotwords"] = hint
        options["initial_prompt"] = hint
    return options


def _transcribe_with_fallback(
    model: Any,
    media_path: Path,
    *,
    language: str | None,
    glossary: str | None,
    on_progress: ProgressCallback | None,
) -> tuple[list[TranscriptSegment], Any]:
    options = decode_options(glossary)
    try:
        return _collect_segments(
            model,
            media_path,
            language=language,
            vad_filter=True,
            options=options,
            on_progress=on_progress,
        )
    except Exception as first_error:
        logger.warning("Retrying transcription without VAD: %s", first_error)
        try:
            return _collect_segments(
                model,
                media_path,
                language=language,
                vad_filter=False,
                options=options,
                on_progress=on_progress,
            )
        except Exception as second_error:
            raise RuntimeError(f"Transcription failed: {second_error}") from second_error


def _collect_segments(
    model: Any,
    media_path: Path,
    *,
    language: str | None,
    vad_filter: bool,
    options: dict[str, Any],
    on_progress: ProgressCallback | None,
) -> tuple[list[TranscriptSegment], Any]:
    segments_iter, info = model.transcribe(
        str(media_path),
        language=language,
        vad_filter=vad_filter,
        **options,
    )
    total = _optional_float(getattr(info, "duration", None)) or 0.0
    collected: list[TranscriptSegment] = []
    for index, raw in enumerate(segments_iter, start=1):
        text = str(getattr(raw, "text", "") or "").strip()
        start = max(0.0, float(getattr(raw, "start", 0.0)))
        end = max(start, float(getattr(raw, "end", start)))
        if text:
            collected.append(TranscriptSegment(start=start, end=end, text=text))
        if total > 0:
            fraction = 0.4 + min(end / total, 1.0) * 0.5
        else:
            fraction = min(0.4 + index * 0.02, 0.9)
        _emit(on_progress, fraction, f"Transcribing {_clock(end)}")
    return collected, info


def _validate_options(
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    estimate_speakers: bool,
    speaker_count: int,
    gap_seconds: float,
) -> None:
    if not _MODEL_NAME.fullmatch(model_name):
        raise ValueError("Whisper model name contains unsupported characters.")
    if device not in DEVICES:
        raise ValueError(f"Device must be one of: {', '.join(DEVICES)}.")
    if compute_type not in COMPUTE_TYPES:
        raise ValueError(f"Compute type must be one of: {', '.join(COMPUTE_TYPES)}.")
    if language is not None and not _LANGUAGE.fullmatch(language):
        raise ValueError("Language must be an ISO code such as en.")
    if estimate_speakers and not 2 <= speaker_count <= 8:
        raise ValueError("Estimated speaker count must be between 2 and 8.")
    if estimate_speakers and not 0 < gap_seconds <= 10:
        raise ValueError("Speaker gap must be between 0 and 10 seconds.")


def _emit(on_progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if on_progress is not None:
        on_progress(max(0.0, min(fraction, 1.0)), message)


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clock(seconds: float) -> str:
    whole = max(0, int(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
