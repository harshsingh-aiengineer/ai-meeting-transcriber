"""Transcript, analysis, and download views."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from collections.abc import Callable
from datetime import datetime

from app.models.schemas import MeetingAnalysis, Transcript
from app.services.export import (
    meeting_to_json,
    meeting_to_markdown,
    transcript_plain_text,
    transcript_timestamped,
    transcript_to_srt,
    transcript_to_text,
)
from app.services.transcription import TranscriptionService
from app.utils.estimates import estimate_processing_seconds
from app.utils.timestamps import format_duration
from app.utils.transcript_cleanup import cleaned_transcript_text
from app.utils.uploads import sanitize_filename


def render_transcript(transcript: Transcript) -> None:
    """Show transcript metadata, the raw transcript, and a cleaned reading view."""
    st.text(f"Source: {Path(transcript.source_name).name}")
    language, duration, segments, model = st.columns(4)
    language.metric("Language", (transcript.language or "Unknown").upper())
    duration.metric("Duration", format_duration(transcript.duration_seconds))
    segments.metric("Segments", str(len(transcript.segments)))
    model.metric("Model", transcript.model)
    if transcript.language_probability is not None:
        st.caption(f"Language confidence: {transcript.language_probability:.0%}")
    if transcript.speaker_mode == "pause_estimate":
        st.caption(
            "Speaker labels are pause-based estimates. They are not voice identification."
        )
    st.caption(
        "Full Transcript is the raw recognition. Cleaned only removes repeated phrases and extra spaces."
    )

    plain = transcript_plain_text(transcript)
    cleaned = cleaned_transcript_text(transcript) or "No speech detected.\n"
    timestamped = transcript_timestamped(transcript)
    srt = transcript_to_srt(transcript) or "No speech detected.\n"
    payload = json.dumps(_segment_document(transcript), indent=2, ensure_ascii=False) + "\n"
    stem = sanitize_filename(Path(transcript.source_name).stem)
    full_tab, cleaned_tab, timed_tab, srt_tab, json_tab = st.tabs(
        ["Full Transcript", "Cleaned", "Timestamped Transcript", "SRT", "JSON"]
    )
    with full_tab:
        st.text_area("Full transcript", value=plain, height=360, disabled=True)
        _download("Download text", plain, f"{stem}-transcript.txt", "text/plain", "download_plain")
    with cleaned_tab:
        st.text_area("Cleaned transcript", value=cleaned, height=360, disabled=True)
        _download(
            "Download cleaned text",
            cleaned,
            f"{stem}-cleaned.txt",
            "text/plain",
            "download_cleaned",
        )
    with timed_tab:
        st.text_area("Timestamped transcript", value=timestamped, height=360, disabled=True)
        _download(
            "Download timestamped text",
            timestamped,
            f"{stem}-timestamped.txt",
            "text/plain",
            "download_timed",
        )
    with srt_tab:
        st.text_area("SRT", value=srt, height=360, disabled=True)
        _download("Download SRT", srt, f"{stem}.srt", "application/x-subrip", "download_srt")
    with json_tab:
        st.text_area("JSON", value=payload, height=360, disabled=True)
        _download("Download JSON", payload, f"{stem}-segments.json", "application/json", "download_segments")


def render_metrics(
    transcript: Transcript | None,
    analysis: MeetingAnalysis | None,
    model: str,
    processing_seconds: float | None,
) -> None:
    """Show the dashboard figures for the current meeting."""
    duration = transcript.duration_seconds if transcript else None
    estimate = estimate_processing_seconds(duration, model)
    segments = len(transcript.segments) if transcript else None
    actions = len(analysis.action_items) if analysis and analysis.available else None
    duration_col, segment_col, time_col, action_col = st.columns(4)
    duration_col.metric("Audio duration", _metric_value(duration, format_duration))
    segment_col.metric("Transcript segments", "—" if segments is None else str(segments))
    time_col.metric("Estimated processing time", _metric_value(estimate, format_duration))
    action_col.metric("Action items", "—" if actions is None else str(actions))
    if processing_seconds is not None:
        st.caption(f"Last run finished in {format_duration(processing_seconds)}.")


def render_summary(analysis: MeetingAnalysis | None, report_date: str | None) -> None:
    """Show the meeting title, date, summary, and related notes."""
    if analysis is None:
        st.info("Transcribe a recording to generate a meeting summary.")
        return
    if not analysis.available:
        st.warning(analysis.message)
        return
    st.markdown(f"**{analysis.title or 'Untitled meeting'}**")
    if report_date:
        st.caption(report_date)
    if analysis.model:
        st.caption(f"Model: {analysis.model}")
    st.write(analysis.summary or "No summary was produced.")
    _text_list("Key points", analysis.key_points)
    _text_list("Risks", analysis.risks)
    _text_list("Follow-up questions asked in the meeting", analysis.follow_up_questions)
    if analysis.suggested_follow_ups:
        st.caption("Suggested follow-ups were not spoken in the meeting.")
        _text_list("Suggested follow-ups", analysis.suggested_follow_ups)


def render_action_items(analysis: MeetingAnalysis | None) -> None:
    """Show action items in a table."""
    if analysis is None:
        st.info("Action items appear after meeting analysis.")
        return
    if not analysis.available:
        st.warning(analysis.message)
        return
    if not analysis.action_items:
        st.text("None identified.")
        return
    st.table(
        [
            {
                "Task": item.task,
                "Owner": item.owner or "—",
                "Deadline": item.deadline or "—",
                "Priority": item.priority or "—",
            }
            for item in analysis.action_items
        ]
    )


def render_decisions(analysis: MeetingAnalysis | None) -> None:
    """Show decisions from the meeting brief."""
    if analysis is None:
        st.info("Decisions appear after meeting analysis.")
        return
    if not analysis.available:
        st.warning(analysis.message)
        return
    if not analysis.decisions:
        st.text("None identified.")
        return
    for index, item in enumerate(analysis.decisions, start=1):
        st.text(f"{index}. {item}")


def render_downloads(
    transcript: Transcript,
    analysis: MeetingAnalysis | None,
    report_date: datetime | None = None,
) -> None:
    """Offer TXT, JSON, SRT, and the Markdown meeting report."""
    stem = sanitize_filename(Path(transcript.source_name).stem)
    text = transcript_to_text(transcript)
    payload = meeting_to_json(transcript, analysis)
    srt = transcript_to_srt(transcript) or "No speech detected.\n"
    markdown = meeting_to_markdown(transcript, analysis, report_date=report_date)
    text_col, json_col, srt_col, report_col = st.columns(4)
    text_col.download_button(
        "TXT",
        data=text.encode("utf-8"),
        file_name=f"{stem}-transcript.txt",
        mime="text/plain",
        key="download_txt",
    )
    json_col.download_button(
        "JSON",
        data=payload.encode("utf-8"),
        file_name=f"{stem}-meeting.json",
        mime="application/json",
        key="download_meeting_json",
    )
    srt_col.download_button(
        "SRT",
        data=srt.encode("utf-8"),
        file_name=f"{stem}.srt",
        mime="application/x-subrip",
        key="download_srt_bundle",
    )
    report_col.download_button(
        "Markdown",
        data=markdown.encode("utf-8"),
        file_name=f"{stem}-report.md",
        mime="text/markdown",
        key="download_md",
    )


def _metric_value(value: float | None, formatter: Callable[[float | None], str]) -> str:
    if value is None:
        return "—"
    return formatter(value)


def _segment_document(transcript: Transcript) -> dict[str, object]:
    service = TranscriptionService(transcript.model or "base", transcript.device or "cpu", transcript.compute_type or "int8")
    return {
        "text": transcript.text,
        "cleaned_text": cleaned_transcript_text(transcript),
        "segments": service.format_segments(transcript.segments),
    }


def _download(label: str, content: str, file_name: str, mime: str, key: str) -> None:
    st.download_button(
        label,
        data=content.encode("utf-8"),
        file_name=file_name,
        mime=mime,
        key=key,
    )


def _text_list(title: str, items: list[str]) -> None:
    st.subheader(title)
    if not items:
        st.text("None identified.")
        return
    for index, item in enumerate(items, start=1):
        st.text(f"{index}. {item}")
