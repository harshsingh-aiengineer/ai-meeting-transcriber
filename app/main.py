"""Streamlit entry point for Meeting Intelligence."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from pydantic import ValidationError

from app.config import ensure_directories, get_settings, log_file
from app.ui.page import render
from app.utils.logging import configure_logging


def main() -> None:
    """Start the Streamlit application."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    st.set_page_config(
        page_title="AI Meeting Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    try:
        settings = get_settings()
    except ValidationError as exc:
        st.error("Configuration is invalid. Check the environment variables in .env.")
        st.code(str(exc))
        return
    ensure_directories()
    configure_logging(settings.log_level, log_file())
    render(settings)


if __name__ == "__main__":
    main()
