"""Sidebar controls for transcription and the optional local LLM."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import streamlit as st

from app.config import (
    CPU_COMPUTE_TYPES,
    CUDA_COMPUTE_TYPES,
    DEVICES,
    WHISPER_MODELS,
    Settings,
)
from app.services.llm import OllamaClient
from app.utils.urls import normalize_base_url

_LANGUAGES: tuple[tuple[str, str | None], ...] = (
    ("Auto-detect", None),
    ("English", "en"),
    ("Hindi", "hi"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Portuguese", "pt"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Arabic", "ar"),
)


@dataclass(frozen=True)
class RunOptions:
    """Options chosen in the sidebar for the current run."""

    whisper_model: str
    device: str
    compute_type: str
    language: str | None
    estimate_speakers: bool
    speaker_count: int
    gap_seconds: float
    ollama_base_url: str
    ollama_model: str
    glossary: str = ""


def render_sidebar(settings: Settings) -> RunOptions:
    """Render settings and return the selected run options."""
    st.sidebar.header("Transcription")
    models = list(WHISPER_MODELS)
    if settings.whisper_model not in models:
        models.insert(0, settings.whisper_model)
    whisper_model = st.sidebar.selectbox(
        "Whisper model",
        models,
        index=models.index(settings.whisper_model),
        key="opt_whisper_model",
        help=(
            "base is the default and is faster on CPU. small recognizes names "
            "and less common words more reliably, and it uses more memory and time. "
            "The model downloads once and stays cached."
        ),
    )
    device = st.sidebar.selectbox(
        "Device",
        DEVICES,
        index=DEVICES.index(settings.device),
        key="opt_device",
    )
    compute_choices = CPU_COMPUTE_TYPES if device == "cpu" else CUDA_COMPUTE_TYPES
    if st.session_state.get("opt_compute") not in compute_choices:
        st.session_state.pop("opt_compute", None)
    preferred = settings.compute_type if settings.compute_type in compute_choices else compute_choices[0]
    compute_type = st.sidebar.selectbox(
        "Compute type",
        compute_choices,
        index=compute_choices.index(preferred),
        key="opt_compute",
        help="int8 is the usual choice on CPU. float16 is for CUDA.",
    )
    language_labels = [label for label, _code in _LANGUAGES]
    language_label = st.sidebar.selectbox(
        "Language",
        language_labels,
        key="opt_language",
        help=(
            "Choose the meeting language when you know it. Auto-detect listens "
            "to the start of the recording and can lock onto the wrong language."
        ),
    )
    language = dict(_LANGUAGES)[language_label]
    glossary = st.sidebar.text_area(
        "Names and terms",
        key="opt_glossary",
        height=88,
        placeholder="Harvey Mackay, Harv Eker, Marci Shimoff, Rick Frishman",
        help=(
            "Optional spellings for people and terms in this recording. "
            "They are hints for recognition. They are not added to the transcript "
            "unless the audio supports them."
        ),
    )

    st.sidebar.header("Speaker turns")
    estimate_speakers = st.sidebar.checkbox(
        "Estimate speakers from pauses",
        value=False,
        key="opt_estimate_speakers",
        help=(
            "Labels switch after a silence. They are not voice identification. "
            "Overlapping speech stays mixed because the audio is converted to mono."
        ),
    )
    speaker_count = int(
        st.sidebar.number_input(
            "Estimated speakers",
            min_value=2,
            max_value=8,
            value=2,
            step=1,
            key="opt_speaker_count",
            disabled=not estimate_speakers,
        )
    )
    gap_seconds = float(
        st.sidebar.slider(
            "Pause that starts a new turn (seconds)",
            min_value=0.6,
            max_value=4.0,
            value=1.2,
            step=0.1,
            key="opt_gap_seconds",
            disabled=not estimate_speakers,
        )
    )

    st.sidebar.header("Local analysis")
    st.sidebar.caption(
        "Summary, decisions, and action items use Ollama. "
        "Transcription still runs when Ollama is unavailable."
    )
    ollama_base_url = st.sidebar.text_input(
        "Ollama base URL",
        value=settings.ollama_base_url,
        key="opt_ollama_url",
    )
    ollama_model = st.sidebar.text_input(
        "Ollama model",
        value=settings.ollama_model,
        key="opt_ollama_model",
    )
    if st.sidebar.button("Check local LLM", key="check_llm"):
        _show_llm_status(ollama_base_url, ollama_model)

    ffmpeg_ready = shutil.which("ffmpeg") and shutil.which("ffprobe")
    st.sidebar.header("Runtime")
    st.sidebar.text("ffmpeg: available" if ffmpeg_ready else "ffmpeg: not found")
    st.sidebar.caption(f"Upload limit: {settings.max_upload_mb} MB")
    return RunOptions(
        whisper_model=str(whisper_model),
        device=str(device),
        compute_type=str(compute_type),
        language=language,
        estimate_speakers=estimate_speakers,
        speaker_count=speaker_count,
        gap_seconds=gap_seconds,
        ollama_base_url=ollama_base_url.strip(),
        ollama_model=ollama_model.strip(),
        glossary=glossary.strip(),
    )


def _show_llm_status(base_url: str, model: str) -> None:
    try:
        client = OllamaClient(normalize_base_url(base_url), model, timeout_seconds=3)
        status = client.health()
    except ValueError as exc:
        st.sidebar.warning(str(exc))
        return
    if status.available:
        st.sidebar.success(status.message)
    else:
        st.sidebar.warning(status.message)
