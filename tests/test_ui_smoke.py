"""Smoke test that the Streamlit entry point can render."""

from pathlib import Path


def test_streamlit_page_renders() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.run()
    assert not app.exception
    labels = [button.label for button in app.button]
    assert "Transcribe" in labels
    assert "Clear" in labels
    assert app.sidebar.selectbox
