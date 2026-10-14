"""Tests for decode settings, cleanup, scoring, and grounded analysis fields."""

from app.models.schemas import Transcript, TranscriptSegment
from app.services.summarization import build_analysis_prompt, merge_analyses, normalize_analysis
from app.services.transcription import decode_options
from app.utils.evaluation import score_transcripts, speaker_overlap_accuracy
from app.utils.glossary import format_glossary
from app.utils.transcript_cleanup import clean_transcript_text, cleaned_transcript_text


def test_decode_options_avoid_repetition_loops_and_keep_speech() -> None:
    options = decode_options("Harvey Mackay, Harvey Mackay")
    assert options["condition_on_previous_text"] is False
    assert options["beam_size"] == 5
    assert options["temperature"] == 0.0
    assert options["vad_parameters"]["threshold"] == 0.35
    assert options["vad_parameters"]["min_speech_duration_ms"] == 0
    assert options["vad_parameters"]["speech_pad_ms"] == 600
    assert options["hotwords"] == "Harvey Mackay"
    assert options["initial_prompt"] == "Harvey Mackay"
    assert "hotwords" not in decode_options("")


def test_glossary_drops_duplicates_and_blank_entries() -> None:
    assert format_glossary(" Harv Eker,\n\nMarci Shimoff; Harv Eker ") == (
        "Harv Eker, Marci Shimoff"
    )
    assert format_glossary("   ") == ""


def test_cleanup_removes_loops_without_adding_words() -> None:
    raw = "the the the report is late the report is late"
    cleaned = clean_transcript_text(raw)
    assert cleaned == "the report is late"
    assert clean_transcript_text("yes yes") == "yes yes"
    raw_words = raw.split()
    for word in cleaned.split():
        assert word.casefold() in {item.casefold() for item in raw_words}


def test_cleaned_view_leaves_raw_segments_unchanged() -> None:
    segment = TranscriptSegment(start=0, end=1, text="go go go now")
    transcript = Transcript(
        meeting_id="a" * 32,
        source_name="demo.wav",
        model="base",
        device="cpu",
        compute_type="int8",
        segments=[segment],
        text=segment.text,
    )
    assert cleaned_transcript_text(transcript) == "go now"
    assert transcript.text == "go go go now"
    assert transcript.segments[0].text == "go go go now"


def test_word_and_character_error_rates() -> None:
    perfect = score_transcripts("ship the report", "Ship the report.")
    assert perfect.wer == 0
    assert perfect.cer == 0
    substitution = score_transcripts("ship the report", "send the report")
    assert substitution.substitutions == 1
    assert substitution.wer == 1 / 3
    omission = score_transcripts("ship the report", "ship report")
    assert omission.deletions == 1
    assert omission.omission_rate == 1 / 3
    extra = score_transcripts("ship the report", "ship the final report")
    assert extra.insertions == 1
    assert extra.insertion_rate == 1 / 3


def test_speaker_overlap_requires_labels() -> None:
    reference = [TranscriptSegment(start=0, end=2, text="Hello", speaker="Speaker 1")]
    unlabeled = [TranscriptSegment(start=0, end=2, text="Hello")]
    assert speaker_overlap_accuracy(reference, unlabeled) is None
    match = [TranscriptSegment(start=0, end=2, text="Hello", speaker="Speaker 1")]
    mismatch = [TranscriptSegment(start=0, end=2, text="Hello", speaker="Speaker 2")]
    assert speaker_overlap_accuracy(reference, match) == 1
    assert speaker_overlap_accuracy(reference, mismatch) == 0


def test_prompt_separates_spoken_questions_from_suggestions() -> None:
    prompt = build_analysis_prompt("Alex asked for the notes.", glossary="Rick Frishman")
    assert "explicitly asked" in prompt
    assert "suggested_follow_ups" in prompt
    assert "Rick Frishman" in prompt
    assert "similar name" in prompt
    plain = build_analysis_prompt("Alex asked for the notes.")
    assert "Preferred spellings" not in plain


def test_suggested_follow_ups_stay_separate_from_spoken_questions() -> None:
    analysis = normalize_analysis(
        {
            "summary": "Done",
            "follow_up_questions": ["Did Alex send the notes?"],
            "suggested_follow_ups": ["Ask for the budget next week."],
        },
        model="qwen2.5:3b",
    )
    merged = merge_analyses([analysis], model="qwen2.5:3b", truncated=False, part_count=1)
    assert merged.follow_up_questions == ["Did Alex send the notes?"]
    assert merged.suggested_follow_ups == ["Ask for the budget next week."]
