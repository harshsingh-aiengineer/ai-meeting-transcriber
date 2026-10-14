"""Tests for dashboard processing estimates."""

from app.utils.estimates import estimate_processing_seconds


def test_estimate_scales_with_model() -> None:
    assert estimate_processing_seconds(10, "tiny") == 2.0
    assert estimate_processing_seconds(10, "base") == 5.0
    assert estimate_processing_seconds(10, "small") == 10.0


def test_estimate_unknown_duration() -> None:
    assert estimate_processing_seconds(None, "base") is None
