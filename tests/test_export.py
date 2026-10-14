"""Tests for transcript formatting and report export."""

from datetime import datetime, timezone

from app.models.schemas import ActionItem, MeetingAnalysis, Transcript, TranscriptSegment
from app.services.export import (
    load_meeting_bundle,
    meeting_to_markdown,
    transcript_plain_text,
    transcript_timestamped,
    transcript_to_srt,
    transcript_to_text,
    write_meeting_bundle,
)


def _transcript() -> Transcript:
    return Transcript(
        meeting_id="abc123",
        source_name="demo.wav",
        language="en",
        duration_seconds=8,
        model="base",
        device="cpu",
        compute_type="int8",
        speaker_mode="pause_estimate",
        segments=[
            TranscriptSegment(
                start=0,
                end=2,
                text="We will ship the report.",
                speaker="Speaker 1",
            ),
            TranscriptSegment(
                start=4,
                end=7,
                text="Alex will send it Friday.",
                speaker="Speaker 2",
            ),
        ],
        text="We will ship the report. Alex will send it Friday.",
    )


def test_transcript_text_includes_timestamps_and_speakers() -> None:
    text = transcript_to_text(_transcript())
    assert "[00:00:00] Speaker 1: We will ship the report." in text
    assert "pause-based estimates" in text


def test_markdown_without_analysis_says_not_run() -> None:
    report = meeting_to_markdown(
        _transcript(),
        None,
        report_date=datetime(2026, 10, 13, tzinfo=timezone.utc),
    )
    assert "## Meeting title" in report
    assert "Untitled meeting" in report
    assert "## Date" in report
    assert "2026-10-13" in report
    assert "## Summary" in report
    assert "Analysis has not been run." in report
    assert "## Follow-up questions" in report


def test_markdown_includes_unavailable_message() -> None:
    analysis = MeetingAnalysis(
        available=False,
        message="Could not reach Ollama at http://localhost:11434.",
    )
    report = meeting_to_markdown(_transcript(), analysis)
    assert "Could not reach Ollama" in report
    assert "Not available." in report


def test_bundle_round_trip(tmp_path) -> None:
    analysis = MeetingAnalysis(
        available=True,
        message="Analysis completed with the local model.",
        summary="The team agreed to ship the report.",
        key_points=["Ship the report"],
        action_items=[ActionItem(task="Send the report", owner="Alex", deadline="Friday", priority="high")],
        decisions=["Ship the report"],
        risks=["Schedule slip"],
        follow_up_questions=["Confirm receipt"],
        model="qwen2.5:3b",
    )
    paths = write_meeting_bundle(tmp_path, _transcript(), analysis)
    loaded, loaded_analysis = load_meeting_bundle(paths.json_path)
    assert loaded.meeting_id == "abc123"
    assert loaded_analysis is not None
    assert loaded_analysis.action_items[0].owner == "Alex"
    assert loaded_analysis.action_items[0].priority == "high"
    assert paths.markdown_path.is_file()
    assert paths.text_path.is_file()
    assert paths.srt_path.is_file()
    assert paths.plain_path.is_file()
    report = paths.markdown_path.read_text(encoding="utf-8")
    assert "## Meeting title" in report
    assert "## Risks" in report
    assert "Schedule slip" in report
    assert "Meeting transcript" in paths.text_path.read_text(encoding="utf-8")
    assert paths.srt_path.read_text(encoding="utf-8").startswith("1\n")


def test_plain_text_has_no_timestamps() -> None:
    plain = transcript_plain_text(_transcript())
    assert "We will ship the report." in plain
    assert "00:00:00" not in plain


def test_timestamped_transcript_and_srt() -> None:
    timestamped = transcript_timestamped(_transcript())
    assert "[00:00:00] Speaker 1: We will ship the report." in timestamped
    srt = transcript_to_srt(_transcript())
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,000\nSpeaker 1: We will ship the report.")
    assert "\n\n2\n00:00:04,000 --> 00:00:07,000\n" in srt
