"""Tests for the DocumentLoader agent."""

import os
import tempfile
import pytest

from agents.document_loader import DocumentLoader, LoadedDocument, SUPPORTED_FORMATS


@pytest.fixture
def loader():
    return DocumentLoader()


@pytest.fixture
def sample_txt(tmp_path):
    """Create a temporary .txt file with sample content."""
    path = tmp_path / "test.txt"
    path.write_text("Hello World.\n\nThis is a test document with two paragraphs.")
    return str(path)


@pytest.fixture
def empty_txt(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    return str(path)


@pytest.fixture
def sample_md(tmp_path):
    """Create a temporary .md file with sample Markdown content."""
    path = tmp_path / "readme.md"
    path.write_text("# Project Title\n\nThis is a **Markdown** document.\n\n## Section\n\n- Item one\n- Item two")
    return str(path)


# --- Loading tests ---

class TestDocumentLoader:
    def test_load_txt_returns_loaded_document(self, loader, sample_txt):
        doc = loader.load(sample_txt)
        assert isinstance(doc, LoadedDocument)

    def test_load_txt_extracts_text(self, loader, sample_txt):
        doc = loader.load(sample_txt)
        assert "Hello World" in doc.text
        assert "two paragraphs" in doc.text

    def test_load_txt_metadata(self, loader, sample_txt):
        doc = loader.load(sample_txt)
        assert doc.filename == "test.txt"
        assert doc.format == "txt"
        assert doc.page_count == 1
        assert doc.char_count > 0

    def test_load_empty_txt(self, loader, empty_txt):
        doc = loader.load(empty_txt)
        assert doc.text == ""
        assert doc.char_count == 0

    def test_file_not_found_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.txt")

    def test_unsupported_format_raises(self, loader, tmp_path):
        path = tmp_path / "test.xlsx"
        path.write_text("data")
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(str(path))

    def test_load_md_extracts_text(self, loader, sample_md):
        doc = loader.load(sample_md)
        assert "Project Title" in doc.text
        assert "**Markdown**" in doc.text
        assert doc.format == "md"
        assert doc.page_count == 1
        assert doc.char_count > 0

    def test_load_md_metadata(self, loader, sample_md):
        doc = loader.load(sample_md)
        assert doc.filename == "readme.md"
        assert doc.format == "md"

    def test_supported_formats_constant(self):
        assert ".pdf" in SUPPORTED_FORMATS
        assert ".docx" in SUPPORTED_FORMATS
        assert ".txt" in SUPPORTED_FORMATS
        assert ".md" in SUPPORTED_FORMATS


# --- Model tests ---

class TestLoadedDocumentModel:
    def test_model_creation(self):
        doc = LoadedDocument(
            filename="test.pdf",
            format="pdf",
            text="Some text",
            page_count=3,
            char_count=9,
        )
        assert doc.filename == "test.pdf"
        assert doc.page_count == 3

    def test_model_json_roundtrip(self):
        doc = LoadedDocument(
            filename="test.txt",
            format="txt",
            text="Hello",
            page_count=1,
            char_count=5,
        )
        json_str = doc.model_dump_json()
        restored = LoadedDocument.model_validate_json(json_str)
        assert restored == doc


# --- Sample file test ---

class TestSampleFile:
    def test_sample_company_report_loads(self, loader):
        sample_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_company_report.txt"
        )
        doc = loader.load(sample_path)
        assert doc.char_count > 1000
        assert "NovaTech" in doc.text
        assert doc.format == "txt"
