"""Tests for the TextChunker agent."""

import pytest

from agents.chunker import TextChunker, TextChunk


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=100, overlap=20)


@pytest.fixture
def long_text():
    """Generate a text that's about 500 characters long."""
    return "".join(f"This is sentence number {i}. " for i in range(20))


# --- Basic chunking ---

class TestChunking:
    def test_chunk_returns_list(self, chunker):
        chunks = chunker.chunk("Hello world", source="test.txt")
        assert isinstance(chunks, list)

    def test_short_text_single_chunk(self, chunker):
        chunks = chunker.chunk("Short text.", source="test.txt")
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, chunker, long_text):
        chunks = chunker.chunk(long_text, source="test.txt")
        assert len(chunks) > 1

    def test_chunks_have_correct_source(self, chunker, long_text):
        chunks = chunker.chunk(long_text, source="report.pdf")
        for chunk in chunks:
            assert chunk.source == "report.pdf"

    def test_chunk_indices_are_sequential(self, chunker, long_text):
        chunks = chunker.chunk(long_text, source="test.txt")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_cover_full_text(self, chunker, long_text):
        chunks = chunker.chunk(long_text, source="test.txt")
        # First chunk starts at 0
        assert chunks[0].start_char == 0
        # Last chunk ends at or near the text length
        assert chunks[-1].end_char >= len(long_text) - chunker.chunk_size

    def test_empty_text_returns_empty_list(self, chunker):
        assert chunker.chunk("", source="test.txt") == []

    def test_whitespace_only_returns_empty(self, chunker):
        assert chunker.chunk("   \n\n  ", source="test.txt") == []


# --- Overlap behavior ---

class TestOverlap:
    def test_chunks_overlap(self):
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A" * 100
        chunks = chunker.chunk(text, source="test.txt")

        # Second chunk should start before first chunk ends
        if len(chunks) >= 2:
            assert chunks[1].start_char < chunks[0].end_char

    def test_no_overlap(self):
        chunker = TextChunker(chunk_size=50, overlap=0)
        text = "A" * 100
        chunks = chunker.chunk(text, source="test.txt")

        if len(chunks) >= 2:
            assert chunks[1].start_char == chunks[0].end_char


# --- Validation ---

class TestValidation:
    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, overlap=-1)

    def test_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, overlap=100)


# --- Model tests ---

class TestTextChunkModel:
    def test_model_creation(self):
        chunk = TextChunk(
            text="Some text",
            chunk_index=0,
            start_char=0,
            end_char=9,
            source="test.txt",
        )
        assert chunk.text == "Some text"

    def test_model_json_roundtrip(self):
        chunk = TextChunk(
            text="Hello",
            chunk_index=2,
            start_char=100,
            end_char=105,
            source="doc.pdf",
        )
        restored = TextChunk.model_validate_json(chunk.model_dump_json())
        assert restored == chunk
