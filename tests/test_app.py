"""Tests for the Streamlit app layer (app.py).

Covers the demo-mode autoload behavior: a fresh session must be able to
answer questions immediately, and clearing all documents must not be
undone by the autoload.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(REPO_DIR, "app.py")
SAMPLE_NAME = "sample_company_report.txt"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """Run the app against an isolated, empty vector store."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    return at


class TestSampleAutoload:
    def test_fresh_session_preloads_the_sample(self, app):
        assert not app.exception
        assert SAMPLE_NAME in app.session_state["uploaded_files"]
        assert app.session_state["total_chunks"] > 0

    def test_question_works_immediately(self, app):
        app.chat_input[0].set_value("What was the company's revenue in 2025?").run()

        assert not app.exception
        roles = [m["role"] for m in app.session_state["messages"]]
        assert roles[-2:] == ["user", "assistant"]

    def test_clear_all_is_not_undone_by_autoload(self, app):
        clear = next(b for b in app.button if "Clear all documents" in b.label)
        clear.click().run()

        assert not app.exception
        assert app.session_state["uploaded_files"] == []
        assert app.session_state["total_chunks"] == 0

    def test_empty_store_error_points_at_the_sample_button(self, app):
        clear = next(b for b in app.button if "Clear all documents" in b.label)
        clear.click().run()

        app.chat_input[0].set_value("Anyone home?").run()

        assert app.error
        assert "Load sample document" in app.error[0].value
