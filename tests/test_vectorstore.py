"""Tests for the VectorStore agent."""

import os
import uuid
import pytest

from agents.chunker import TextChunk
from agents.vectorstore import VectorStore, SearchResult


@pytest.fixture
def store():
    """Create a fresh in-memory vector store with a unique collection name."""
    return VectorStore(collection_name=f"test_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def sample_chunks():
    """Create sample chunks about different topics."""
    return [
        TextChunk(
            text="The company revenue was EUR 47.3 million in 2025, up 31% year over year.",
            chunk_index=0, start_char=0, end_char=72, source="report.txt",
        ),
        TextChunk(
            text="The engineering team grew from 50 to 85 people during 2025.",
            chunk_index=1, start_char=72, end_char=130, source="report.txt",
        ),
        TextChunk(
            text="The main product is a cloud-based supply chain optimization platform.",
            chunk_index=2, start_char=130, end_char=198, source="report.txt",
        ),
        TextChunk(
            text="Key competitors include Kinaxis, o9 Solutions, and Blue Yonder.",
            chunk_index=3, start_char=198, end_char=261, source="report.txt",
        ),
        TextChunk(
            text="The company plans to expand into the UK and Benelux markets in 2026.",
            chunk_index=4, start_char=261, end_char=329, source="report.txt",
        ),
    ]


# --- Basic operations ---

class TestBasicOperations:
    def test_empty_store_has_zero_count(self, store):
        assert store.count == 0

    def test_add_chunks_increases_count(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        assert store.count == 5

    def test_add_empty_list_returns_zero(self, store):
        assert store.add_chunks([]) == 0

    def test_add_returns_chunk_count(self, store, sample_chunks):
        result = store.add_chunks(sample_chunks)
        assert result == 5


# --- Search ---

class TestSearch:
    def test_search_returns_results(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("How much revenue?")
        assert len(results) > 0

    def test_search_returns_search_result_objects(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("revenue")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_revenue_query_finds_revenue_chunk(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("How much money did the company make?", top_k=2)
        texts = [r.text for r in results]
        assert any("revenue" in t.lower() or "47.3" in t for t in texts)

    def test_team_query_finds_team_chunk(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("engineering team grew people", top_k=2)
        texts = [r.text for r in results]
        assert any("team" in t.lower() or "grew" in t.lower() for t in texts)

    def test_search_respects_top_k(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("company", top_k=3)
        assert len(results) <= 3

    def test_search_empty_store_returns_empty(self, store):
        results = store.search("anything")
        assert results == []

    def test_search_result_has_source(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("revenue")
        for r in results:
            assert r.source == "report.txt"

    def test_search_result_has_distance(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("revenue")
        for r in results:
            assert isinstance(r.distance, float)


# --- Reset ---

class TestReset:
    def test_reset_clears_store(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        assert store.count == 5
        store.reset()
        assert store.count == 0


# --- Multiple documents ---

class TestMultipleDocuments:
    def test_add_from_multiple_sources(self, store):
        chunks_a = [
            TextChunk(text="Revenue is 100M", chunk_index=0,
                      start_char=0, end_char=15, source="doc_a.pdf"),
        ]
        chunks_b = [
            TextChunk(text="Profit is 20M", chunk_index=0,
                      start_char=0, end_char=13, source="doc_b.pdf"),
        ]
        store.add_chunks(chunks_a)
        store.add_chunks(chunks_b)
        assert store.count == 2

    def test_search_returns_correct_source(self, store):
        store.add_chunks([
            TextChunk(text="Python is a programming language",
                      chunk_index=0, start_char=0, end_char=31, source="python.txt"),
            TextChunk(text="Java is used for enterprise applications",
                      chunk_index=0, start_char=0, end_char=40, source="java.txt"),
        ])
        results = store.search("Python programming", top_k=1)
        assert results[0].source == "python.txt"


# --- Model tests ---

class TestSearchResultModel:
    def test_model_creation(self):
        r = SearchResult(
            text="Some text",
            source="file.pdf",
            chunk_index=0,
            distance=0.123,
        )
        assert r.distance == 0.123

    def test_model_json_roundtrip(self):
        r = SearchResult(
            text="Test", source="f.txt", chunk_index=1, distance=0.5,
        )
        restored = SearchResult.model_validate_json(r.model_dump_json())
        assert restored == r


# --- Persistence ---

class TestPersistence:
    def test_data_survives_reopen(self, tmp_path, sample_chunks):
        persist_dir = str(tmp_path / "chroma_store")
        col_name = f"test_{uuid.uuid4().hex[:8]}"

        store1 = VectorStore(persist_dir=persist_dir, collection_name=col_name)
        store1.add_chunks(sample_chunks)
        assert store1.count == 5

        store2 = VectorStore(persist_dir=persist_dir, collection_name=col_name)
        assert store2.count == 5

    def test_search_works_after_reopen(self, tmp_path, sample_chunks):
        persist_dir = str(tmp_path / "chroma_store")
        col_name = f"test_{uuid.uuid4().hex[:8]}"

        store1 = VectorStore(persist_dir=persist_dir, collection_name=col_name)
        store1.add_chunks(sample_chunks)

        store2 = VectorStore(persist_dir=persist_dir, collection_name=col_name)
        results = store2.search("revenue", top_k=2)
        assert len(results) > 0
        assert any("revenue" in r.text.lower() for r in results)

    def test_persistent_store_creates_directory(self, tmp_path):
        persist_dir = str(tmp_path / "new_dir" / "chroma")
        VectorStore(persist_dir=persist_dir)
        assert os.path.isdir(persist_dir)

    def test_reset_clears_persistent_store(self, tmp_path, sample_chunks):
        persist_dir = str(tmp_path / "chroma_store")
        col_name = f"test_{uuid.uuid4().hex[:8]}"

        store = VectorStore(persist_dir=persist_dir, collection_name=col_name)
        store.add_chunks(sample_chunks)
        store.reset()
        assert store.count == 0


# --- List sources ---

class TestListSources:
    def test_empty_store_returns_empty(self, store):
        assert store.list_sources() == []

    def test_returns_unique_sources(self, store):
        chunks = [
            TextChunk(text="A", chunk_index=0, start_char=0, end_char=1, source="a.pdf"),
            TextChunk(text="B", chunk_index=1, start_char=1, end_char=2, source="a.pdf"),
            TextChunk(text="C", chunk_index=0, start_char=0, end_char=1, source="b.txt"),
        ]
        store.add_chunks(chunks)
        sources = store.list_sources()
        assert sources == ["a.pdf", "b.txt"]

    def test_sources_sorted(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        extra = [TextChunk(text="X", chunk_index=0, start_char=0, end_char=1, source="aaa.txt")]
        store.add_chunks(extra)
        sources = store.list_sources()
        assert sources == sorted(sources)
