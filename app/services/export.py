"""Text, Markdown, and JSON exports for transcripts and reports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.schemas import ActionItem, MeetingAnalysis, Transcript
from app.utils.timestamps import format_duration, format_srt_timestamp, format_timestamp
from app.utils.transcript_cleanup import cleaned_transcript_text


@dataclass(frozen=True)
class ExportedPaths:
    """Files written for one meeting."""

    json_path: Path
    markdown_path: Path
    text_path: Path
    srt_path: Path
    plain_path: Path


def transcript_plain_text(transcript: Transcript) -> str:
    """Return the spoken text without timestamps."""
    parts = [segment.text.strip() for segment in transcript.segments if segment.text.strip()]
    if parts:
        return "\n\n".join(parts) + "\n"
    if transcript.text.strip():
        return transcript.text.strip() + "\n"
    return "No speech detected.\n"


def transcript_timestamped(transcript: Transcript) -> str:
    """Return one timestamped line per segment."""
    lines = [
        _segment_line(segment.start, segment.speaker, segment.text)
        for segment in transcript.segments
    ]
    if not lines:
        return "No speech detected.\n"
    return "\n".join(lines) + "\n"


def transcript_to_srt(transcript: Transcript) -> str:
    """Render segments as an SRT subtitle file."""
    blocks: list[str] = []
    for index, segment in enumerate(transcript.segments, start=1):
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        blocks.append(
            "\n".join(
                [
                    str(index),
                    (
                        f"{format_srt_timestamp(segment.start)} --> "
                        f"{format_srt_timestamp(segment.end)}"
                    ),
                    f"{speaker}{segment.text}",
                ]
            )
        )
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def transcript_to_text(transcript: Transcript) -> str:
    """Render a standalone plain-text transcript."""
    lines = [
        "Meeting transcript",
        f"Source: {transcript.source_name}",
        f"Model: {transcript.model}",
        f"Language: {transcript.language or 'Unknown'}",
        f"Duration: {format_duration(transcript.duration_seconds)}",
        f"Speaker labels: {_speaker_label(transcript)}",
        "",
    ]
    body = [
        _segment_line(segment.start, segment.speaker, segment.text)
        for segment in transcript.segments
    ]
    if not body:
        body.append("No speech detected.")
    return "\n".join(lines + body) + "\n"


def meeting_to_markdown(
    transcript: Transcript,
    analysis: MeetingAnalysis | None,
    report_date: datetime | None = None,
) -> str:
    """Render a full meeting report in Markdown."""
    summary, key_points, actions, decisions, risks, questions, suggestions = _analysis_sections(
        analysis
    )
    title = _meeting_title(analysis)
    lines = [
        f"# {title}",
        "",
        "## Meeting title",
        "",
        title,
        "",
        "## Date",
        "",
        _report_date(report_date),
        "",
        f"- Source: {transcript.source_name}",
        f"- Model: {transcript.model}",
        f"- Device: {transcript.device}",
        f"- Language: {transcript.language or 'Unknown'}",
        f"- Duration: {format_duration(transcript.duration_seconds)}",
        f"- Speaker labels: {_speaker_label(transcript)}",
        "",
        "## Summary",
        "",
        summary,
        "",
        "## Key points",
        "",
        *key_points,
        "",
        "## Decisions",
        "",
        *decisions,
        "",
        "## Action items",
        "",
        *actions,
        "",
        "## Risks",
        "",
        *risks,
        "",
        "## Follow-up questions",
        "",
        *questions,
        "",
        "## Suggested follow-ups",
        "",
        "These questions were not spoken in the meeting.",
        "",
        *suggestions,
        "",
        "## Transcript",
        "",
        transcript_timestamped(transcript).rstrip(),
        "",
        "## Cleaned transcript",
        "",
        "Repeated phrases and extra spaces are removed. No words are added.",
        "",
        cleaned_transcript_text(transcript) or "No speech detected.",
        "",
    ]
    return "\n".join(lines)


def meeting_document(
    transcript: Transcript,
    analysis: MeetingAnalysis | None,
) -> dict[str, Any]:
    """Return the JSON document stored on disk and offered for download."""
    return {
        "transcript": transcript.model_dump(mode="json"),
        "analysis": analysis.model_dump(mode="json") if analysis else None,
    }


def meeting_to_json(transcript: Transcript, analysis: MeetingAnalysis | None) -> str:
    """Serialize a meeting document."""
    return json.dumps(
        meeting_document(transcript, analysis),
        indent=2,
        ensure_ascii=False,
    ) + "\n"


def write_meeting_bundle(
    output_directory: Path,
    transcript: Transcript,
    analysis: MeetingAnalysis | None,
    report_date: datetime | None = None,
) -> ExportedPaths:
    """Write JSON, Markdown, SRT, and plain-text files for a meeting."""
    meeting_id = _require_meeting_id(transcript.meeting_id)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = ExportedPaths(
        json_path=output_directory / f"{meeting_id}.json",
        markdown_path=output_directory / f"{meeting_id}.md",
        text_path=output_directory / f"{meeting_id}.txt",
        srt_path=output_directory / f"{meeting_id}.srt",
        plain_path=output_directory / f"{meeting_id}.plain.txt",
    )
    paths.json_path.write_text(
        meeting_to_json(transcript, analysis),
        encoding="utf-8",
    )
    paths.markdown_path.write_text(
        meeting_to_markdown(transcript, analysis, report_date=report_date),
        encoding="utf-8",
    )
    paths.text_path.write_text(transcript_to_text(transcript), encoding="utf-8")
    paths.srt_path.write_text(transcript_to_srt(transcript), encoding="utf-8")
    paths.plain_path.write_text(transcript_plain_text(transcript), encoding="utf-8")
    return paths


def load_meeting_bundle(path: Path) -> tuple[Transcript, MeetingAnalysis | None]:
    """Load a meeting JSON bundle from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "transcript" not in payload:
        raise ValueError("Saved meeting file is missing a transcript.")
    transcript = Transcript.model_validate(payload["transcript"])
    raw_analysis = payload.get("analysis")
    analysis = (
        MeetingAnalysis.model_validate(raw_analysis)
        if raw_analysis is not None
        else None
    )
    return transcript, analysis


def _analysis_sections(
    analysis: MeetingAnalysis | None,
) -> tuple[str, list[str], list[str], list[str], list[str], list[str], list[str]]:
    if analysis is None:
        placeholder = ["Analysis has not been run."]
        return (
            "Analysis has not been run.",
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
        )
    if not analysis.available:
        placeholder = ["Not available."]
        return (
            analysis.message,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
        )
    summary = analysis.summary or "No summary was produced."
    return (
        summary,
        _bullets(analysis.key_points),
        _action_lines(analysis.action_items),
        _bullets(analysis.decisions),
        _bullets(analysis.risks),
        _bullets(analysis.follow_up_questions),
        _bullets(analysis.suggested_follow_ups),
    )


def _bullets(items: list[str]) -> list[str]:
    if not items:
        return ["None identified."]
    return [f"- {item}" for item in items]


def _action_lines(items: list[ActionItem]) -> list[str]:
    if not items:
        return ["None identified."]
    lines: list[str] = []
    for item in items:
        details = ""
        if item.owner:
            details += f" (owner: {item.owner})"
        if item.deadline:
            details += f" (deadline: {item.deadline})"
        if item.priority:
            details += f" (priority: {item.priority})"
        lines.append(f"- {item.task}{details}")
    return lines


def _segment_line(start: float, speaker: str | None, text: str) -> str:
    label = f"{speaker}: " if speaker else ""
    return f"[{format_timestamp(start)}] {label}{text}"


def _require_meeting_id(meeting_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", meeting_id):
        raise ValueError("Meeting id contains unsupported characters.")
    return meeting_id


def _meeting_title(analysis: MeetingAnalysis | None) -> str:
    if analysis is not None and analysis.title.strip():
        return analysis.title.strip()
    return "Untitled meeting"


def _report_date(report_date: datetime | None) -> str:
    when = report_date or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _speaker_label(transcript: Transcript) -> str:
    if transcript.speaker_mode == "pause_estimate":
        return "pause-based estimates, not voice identification"
    return "not included"
