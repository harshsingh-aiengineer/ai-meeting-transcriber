"""Streamlit page composition for Meeting Intelligence."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from app.config import Settings, database_path, output_dir, upload_dir
from app.models.schemas import MeetingAnalysis, MeetingRecord, Transcript
from app.services.export import load_meeting_bundle, write_meeting_bundle
from app.services.llm import OLLAMA_UNAVAILABLE_MESSAGE
from app.services.meeting_analysis import MeetingAnalysisService
from app.services.storage import MeetingStore
from app.services.transcription import TranscriptionService
from app.ui.results import (
    render_action_items,
    render_decisions,
    render_downloads,
    render_metrics,
    render_summary,
    render_transcript,
)
from app.ui.sidebar import RunOptions, render_sidebar
from app.utils.media import MediaError
from app.utils.timestamps import format_duration
from app.utils.uploads import (
    redact_user_message,
    resolve_under_directory,
    save_upload,
    upload_type_labels,
)

logger = logging.getLogger(__name__)


def render(settings: Settings) -> None:
    """Render the full Meeting Intelligence page."""
    _init_state()
    options = render_sidebar(settings)
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d7e0ea;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Meeting Intelligence")
    st.caption(
        "Upload a recording, transcribe it locally, and export a structured meeting report."
    )
    dashboard = st.empty()
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        st.warning(
            "ffmpeg was not found on PATH. Install ffmpeg and restart the app before transcribing."
        )

    with st.container(border=True):
        st.subheader("Upload")
        uploaded = st.file_uploader(
            "Audio or video file",
            type=upload_type_labels(),
            help=(
                f"Maximum size {settings.max_upload_mb} MB. "
                f"Supported: {', '.join(upload_type_labels())}."
            ),
            key=f"upload_{st.session_state.uploader_key}",
        )
        transcribe_col, clear_col = st.columns([1, 1])
        transcribe_clicked = transcribe_col.button(
            "Transcribe",
            type="primary",
            disabled=uploaded is None,
            key="transcribe_button",
        )
        clear_clicked = clear_col.button("Clear", key="clear_button")
    if clear_clicked:
        _clear_state()
        st.rerun()
    if transcribe_clicked and uploaded is not None:
        _run_transcription(uploaded, options, settings)

    if st.session_state.error:
        st.error(st.session_state.error)

    transcript = _current_transcript()
    analysis = _current_analysis()
    report_date = _report_date()
    with dashboard.container(border=True):
        st.subheader("Dashboard")
        render_metrics(
            transcript,
            analysis,
            options.whisper_model,
            st.session_state.get("processing_seconds"),
        )
    with st.container(border=True):
        st.subheader("Transcription")
        if transcript is None:
            st.info("Upload a recording and choose Transcribe. The transcript will appear here.")
            _empty_transcript_tabs()
        else:
            render_transcript(transcript)
    with st.container(border=True):
        st.subheader("Meeting Summary")
        _render_analysis_controls(transcript, analysis, options, settings)
        render_summary(analysis, report_date.strftime("%Y-%m-%d") if report_date else None)
    with st.container(border=True):
        st.subheader("Action Items")
        render_action_items(analysis)
    with st.container(border=True):
        st.subheader("Decisions")
        render_decisions(analysis)
    with st.container(border=True):
        st.subheader("Downloads")
        if transcript is None:
            st.info("TXT, JSON, SRT, and Markdown exports appear after transcription.")
        else:
            render_downloads(transcript, analysis, report_date=report_date)
    with st.expander("History"):
        _render_history()


def _render_analysis_controls(
    transcript: Transcript | None,
    analysis: MeetingAnalysis | None,
    options: RunOptions,
    settings: Settings,
) -> None:
    if transcript is None:
        return
    run_analysis = st.button(
        "Run analysis again" if analysis is not None else "Analyze meeting",
        type="primary",
        key="analyze_button",
    )
    if run_analysis:
        with st.spinner("Checking Ollama, then reading the transcript..."):
            analysis = MeetingAnalysisService(
                base_url=options.ollama_base_url,
                model=options.ollama_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            ).analyze(transcript, glossary=options.glossary)
        st.session_state.analysis = analysis.model_dump(mode="json")
        _persist_analysis(transcript, analysis)


def _run_transcription(uploaded: Any, options: RunOptions, settings: Settings) -> None:
    meeting_id = uuid.uuid4().hex
    source_name = Path(uploaded.name or "upload").name
    st.session_state.error = None
    store = _open_store()
    if store is not None:
        try:
            store.create(
                MeetingRecord(
                    id=meeting_id,
                    filename=source_name,
                    created_at=datetime.now(timezone.utc),
                    whisper_model=options.whisper_model,
                    status="processing",
                )
            )
        except sqlite3.Error:
            logger.exception("Could not create meeting row meeting_id=%s", meeting_id)
            store = None

    status_slot = st.empty()
    progress = st.progress(0.0)
    started = time.perf_counter()
    st.session_state.report_date = datetime.now(timezone.utc).isoformat()
    try:
        status_slot.info("Saving upload.")
        saved_path = save_upload(
            uploaded.getvalue(),
            source_name,
            upload_dir(),
            meeting_id,
            settings.max_upload_mb * 1024 * 1024,
        )

        def on_progress(fraction: float, message: str) -> None:
            progress.progress(fraction)
            status_slot.info(message)

        service = TranscriptionService(
            options.whisper_model,
            options.device,
            options.compute_type,
        )
        transcript = service.transcribe(
            saved_path,
            meeting_id=meeting_id,
            source_name=source_name,
            language=options.language,
            estimate_speakers=options.estimate_speakers,
            speaker_count=options.speaker_count,
            gap_seconds=options.gap_seconds,
            glossary=options.glossary,
            on_progress=on_progress,
        )
        service.save_transcript(transcript, output_dir() / f"{meeting_id}.segments.json")
        paths = write_meeting_bundle(
            output_dir(),
            transcript,
            None,
            report_date=_report_date(),
        )
        if store is not None:
            store.update(
                meeting_id,
                status="completed",
                language=transcript.language,
                duration_seconds=transcript.duration_seconds,
                transcript_path=str(paths.json_path),
                report_path=str(paths.markdown_path),
                error=None,
            )
        st.session_state.transcript = transcript.model_dump(mode="json")
        st.session_state.meeting_id = meeting_id
        st.session_state.processing_seconds = round(time.perf_counter() - started, 1)
        progress.progress(1.0)
        status_slot.info("Checking whether local meeting analysis is available.")
        analysis_failed = False
        try:
            analysis = MeetingAnalysisService(
                base_url=options.ollama_base_url,
                model=options.ollama_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            ).analyze(transcript, glossary=options.glossary)
        except Exception:
            logger.exception("Analysis failed after transcription meeting_id=%s", meeting_id)
            analysis = MeetingAnalysis(available=False, message=OLLAMA_UNAVAILABLE_MESSAGE)
            analysis_failed = True
        _persist_analysis(transcript, analysis)
        if analysis_failed:
            _mark_analysis(meeting_id, "failed")
        st.session_state.analysis = analysis.model_dump(mode="json")
        status_slot.success("Transcript ready.")
    except MediaError:
        logger.exception("Unreadable recording meeting_id=%s", meeting_id)
        progress.empty()
        status_slot.empty()
        _mark_failed(
            store,
            meeting_id,
            "The recording could not be read. Use a valid audio or video file.",
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("Transcription failed meeting_id=%s", meeting_id)
        progress.empty()
        status_slot.empty()
        _mark_failed(store, meeting_id, redact_user_message(str(exc)))
    except Exception:
        logger.exception("Unexpected transcription failure meeting_id=%s", meeting_id)
        progress.empty()
        status_slot.empty()
        _mark_failed(store, meeting_id, "Transcription failed. See the application log for details.")


def _persist_analysis(transcript: Transcript, analysis: MeetingAnalysis) -> None:
    try:
        paths = write_meeting_bundle(
            output_dir(),
            transcript,
            analysis,
            report_date=_report_date(),
        )
    except OSError:
        logger.exception("Could not write analysis files meeting_id=%s", transcript.meeting_id)
        st.warning("Analysis is shown here, but it could not be written to the outputs folder.")
        return
    store = _open_store()
    if store is None:
        return
    try:
        store.update(
            transcript.meeting_id,
            transcript_path=str(paths.json_path),
            report_path=str(paths.markdown_path),
            analysis_status="completed" if analysis.available else "unavailable",
        )
    except sqlite3.Error:
        logger.exception("Could not update analysis paths meeting_id=%s", transcript.meeting_id)


def _render_history() -> None:
    store = _open_store()
    if store is None:
        st.warning("Meeting history is unavailable because the local database could not be opened.")
        return
    try:
        records = store.list_recent(limit=10)
    except sqlite3.Error:
        logger.exception("Could not list meetings")
        st.warning("Meeting history could not be read.")
        return
    if not records:
        st.info("Completed meetings will be listed here.")
        return
    for record in records:
        name_col, meta_col, open_col = st.columns([3, 2, 1])
        name_col.text(record.filename)
        if record.error:
            name_col.text(redact_user_message(record.error)[:180])
        created = record.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        analysis_label = record.analysis_status or "pending"
        meta_col.text(
            f"{record.status} · analysis {analysis_label} · "
            f"{format_duration(record.duration_seconds)} · {created}"
        )
        if record.transcript_path and open_col.button("Open", key=f"open_{record.id}"):
            _open_record(record)


def _open_record(record: MeetingRecord) -> None:
    if not record.transcript_path:
        st.session_state.error = "This meeting does not have a saved transcript."
        return
    try:
        path = resolve_under_directory(output_dir(), Path(record.transcript_path))
        transcript, analysis = load_meeting_bundle(path)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.exception("Could not open meeting_id=%s", record.id)
        st.session_state.error = redact_user_message(f"Could not open the saved meeting. {exc}")
        return
    st.session_state.transcript = transcript.model_dump(mode="json")
    st.session_state.analysis = analysis.model_dump(mode="json") if analysis else None
    st.session_state.meeting_id = transcript.meeting_id
    st.session_state.error = None
    st.rerun()


def _open_store() -> MeetingStore | None:
    try:
        return MeetingStore(database_path())
    except sqlite3.Error:
        logger.exception("Could not open the meeting database")
        return None


def _mark_analysis(meeting_id: str, analysis_status: str) -> None:
    store = _open_store()
    if store is None:
        return
    try:
        store.update(meeting_id, analysis_status=analysis_status)
    except sqlite3.Error:
        logger.exception("Could not update analysis status meeting_id=%s", meeting_id)


def _report_date() -> datetime | None:
    raw = st.session_state.get("report_date")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _mark_failed(store: MeetingStore | None, meeting_id: str, message: str) -> None:
    st.session_state.error = message
    st.session_state.transcript = None
    st.session_state.analysis = None
    if store is None:
        return
    try:
        store.update(meeting_id, status="failed", error=message[:500])
    except sqlite3.Error:
        logger.exception("Could not record the transcription failure meeting_id=%s", meeting_id)


def _empty_transcript_tabs() -> None:
    full_tab, cleaned_tab, timed_tab, srt_tab, json_tab = st.tabs(
        ["Full Transcript", "Cleaned", "Timestamped Transcript", "SRT", "JSON"]
    )
    for tab in (full_tab, cleaned_tab, timed_tab, srt_tab, json_tab):
        with tab:
            st.caption("No transcript yet.")


def _init_state() -> None:
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("meeting_id", None)
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("processing_seconds", None)
    st.session_state.setdefault("report_date", None)


def _clear_state() -> None:
    st.session_state.transcript = None
    st.session_state.analysis = None
    st.session_state.error = None
    st.session_state.meeting_id = None
    st.session_state.processing_seconds = None
    st.session_state.report_date = None
    st.session_state.uploader_key += 1


def _current_transcript() -> Transcript | None:
    raw = st.session_state.get("transcript")
    if not raw:
        return None
    return Transcript.model_validate(raw)


def _current_analysis() -> MeetingAnalysis | None:
    raw = st.session_state.get("analysis")
    if not raw:
        return None
    return MeetingAnalysis.model_validate(raw)
