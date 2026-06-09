"""Tests for the QA Agent."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from agents.chunker import TextChunk
from agents.qa_agent import SYSTEM_PROMPT, QAAgent, QAResponse, Source
from agents.vectorstore import VectorStore


@pytest.fixture
def mock_store():
    """Create a vector store with sample data."""
    store = VectorStore(collection_name=f"test_{uuid.uuid4().hex[:8]}")
    store.add_chunks(
        [
            TextChunk(
                text="The company revenue was EUR 47.3 million in 2025.",
                chunk_index=0,
                start_char=0,
                end_char=50,
                source="report.txt",
            ),
            TextChunk(
                text="The CEO is Dr. Sarah Müller, a former McKinsey partner.",
                chunk_index=1,
                start_char=50,
                end_char=105,
                source="report.txt",
            ),
            TextChunk(
                text="Main competitors are Kinaxis and o9 Solutions.",
                chunk_index=2,
                start_char=105,
                end_char=151,
                source="report.txt",
            ),
        ]
    )
    return store


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="The revenue was EUR 47.3 million in 2025.")]
    return mock_response


# --- QA Response model ---


class TestQAResponseModel:
    def test_model_creation(self):
        resp = QAResponse(
            answer="The revenue is 47M.",
            sources=[
                Source(text="Revenue was 47M", source="report.txt", chunk_index=0),
            ],
            model="claude-sonnet-4-6",
        )
        assert resp.answer == "The revenue is 47M."
        assert len(resp.sources) == 1

    def test_model_empty_sources(self):
        resp = QAResponse(answer="No info found.", sources=[], model="claude-sonnet-4-6")
        assert resp.sources == []

    def test_model_json_roundtrip(self):
        resp = QAResponse(
            answer="Answer",
            sources=[Source(text="src", source="f.txt", chunk_index=0)],
            model="claude-sonnet-4-6",
        )
        restored = QAResponse.model_validate_json(resp.model_dump_json())
        assert restored == resp


# --- Source model ---


class TestSourceModel:
    def test_source_creation(self):
        src = Source(text="Some text", source="doc.pdf", chunk_index=3)
        assert src.chunk_index == 3

    def test_source_truncation_in_response(self):
        """Sources in QAResponse should be readable even with long text."""
        long_text = "A" * 500
        src = Source(text=long_text, source="doc.pdf", chunk_index=0)
        assert len(src.text) == 500


# --- QA Agent ---


class TestQAAgent:
    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_ask_returns_qa_response(self, mock_client_class, mock_store, mock_anthropic_response):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        agent = QAAgent(vector_store=mock_store, api_key="test-key")
        response = agent.ask("What is the revenue?")

        assert isinstance(response, QAResponse)
        assert "47.3 million" in response.answer

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_ask_includes_sources(self, mock_client_class, mock_store, mock_anthropic_response):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        agent = QAAgent(vector_store=mock_store, api_key="test-key")
        response = agent.ask("What is the revenue?")

        assert len(response.sources) > 0
        assert all(isinstance(s, Source) for s in response.sources)

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_ask_sends_context_to_api(self, mock_client_class, mock_store, mock_anthropic_response):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        agent = QAAgent(vector_store=mock_store, api_key="test-key")
        agent.ask("What is the revenue?")

        # Verify the API was called
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        # Check that system prompt is passed
        assert call_kwargs.kwargs["system"] == SYSTEM_PROMPT

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_ask_empty_store(self, mock_client_class):
        empty_store = VectorStore(collection_name=f"empty_{uuid.uuid4().hex[:8]}")
        agent = QAAgent(vector_store=empty_store, api_key="test-key")
        response = agent.ask("Anything?")

        assert "upload" in response.answer.lower() or "no documents" in response.answer.lower()
        assert response.sources == []

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_ask_respects_top_k(self, mock_client_class, mock_store, mock_anthropic_response):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        agent = QAAgent(vector_store=mock_store, api_key="test-key", top_k=2)
        response = agent.ask("Tell me about the company")

        assert len(response.sources) <= 2

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_build_context_format(self, mock_client_class, mock_store, mock_anthropic_response):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        agent = QAAgent(vector_store=mock_store, api_key="test-key")
        results = mock_store.search("revenue", top_k=2)
        context = agent._build_context(results)

        assert "[Chunk 1 |" in context
        assert "Source:" in context


# --- Conversation memory ---


class TestConversationHistory:
    """ask() must forward previous turns to the LLM and cap history length."""

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_no_history_sends_single_user_message(
        self, mock_client_class, mock_store, mock_anthropic_response
    ):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        agent = QAAgent(vector_store=mock_store, api_key="test-key")
        agent.ask("Who is the CEO?")

        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_history_is_forwarded(self, mock_client_class, mock_store, mock_anthropic_response):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        agent = QAAgent(vector_store=mock_store, api_key="test-key")
        history = [
            {"role": "user", "content": "What is the revenue?"},
            {"role": "assistant", "content": "EUR 47.3 million."},
        ]
        agent.ask("And the year before?", conversation_history=history)

        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "What is the revenue?"}
        assert messages[1] == {"role": "assistant", "content": "EUR 47.3 million."}
        assert messages[2]["role"] == "user"
        assert "year before" in messages[2]["content"]

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_history_is_trimmed_to_max_turns(
        self, mock_client_class, mock_store, mock_anthropic_response
    ):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        # 6 full turns = 12 messages; max=2 turns should keep the last 4
        history = []
        for i in range(6):
            history.append({"role": "user", "content": f"q{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})

        agent = QAAgent(vector_store=mock_store, api_key="test-key", max_history_turns=2)
        agent.ask("latest?", conversation_history=history)

        messages = mock_client.messages.create.call_args.kwargs["messages"]
        # 2 turns * 2 messages + 1 new user = 5
        assert len(messages) == 5
        assert messages[0] == {"role": "user", "content": "q4"}
        assert messages[1] == {"role": "assistant", "content": "a4"}
        assert messages[-1]["role"] == "user"
        assert "latest" in messages[-1]["content"]

    @patch("agents.qa_agent.anthropic.Anthropic")
    def test_history_filters_unexpected_roles(
        self, mock_client_class, mock_store, mock_anthropic_response
    ):
        mock_client = mock_client_class.return_value
        mock_client.messages.create.return_value = mock_anthropic_response

        history = [
            {"role": "user", "content": "ok"},
            {"role": "system", "content": "ignored"},  # should be dropped
            {"role": "assistant", "content": "ok back"},
            {"role": "user"},  # missing content
            {"foo": "bar"},  # not a turn
        ]
        agent = QAAgent(vector_store=mock_store, api_key="test-key")
        agent.ask("next?", conversation_history=history)

        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]

    def test_demo_mode_ignores_history(self, mock_store):
        agent = QAAgent(vector_store=mock_store, demo_mode=True)
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "answer"},
        ]

        # Should not raise even though no API client exists, and should
        # still produce a demo-mode answer with the retrieved chunks.
        response = agent.ask("revenue?", conversation_history=history)
        assert response.model == "demo-mode"
        assert "DEMO MODE" in response.answer


# --- System prompt ---


class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_context(self):
        assert "context" in SYSTEM_PROMPT.lower()

    def test_system_prompt_limits_to_documents(self):
        assert "only" in SYSTEM_PROMPT.lower()
