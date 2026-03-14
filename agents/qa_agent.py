"""
QA Agent — Answers questions about documents using RAG.

RAG = Retrieval-Augmented Generation
=====================================
This is the heart of the system. When a user asks a question:

  1. RETRIEVE: Search the vector store for relevant text chunks
  2. AUGMENT:  Build a prompt that includes those chunks as context
  3. GENERATE: Send the prompt to the LLM, which answers based on the context

WHY NOT JUST SEND THE WHOLE DOCUMENT?
  - Documents can be very long (thousands of pages)
  - AI models have a limited context window
  - It's expensive to send large amounts of text
  - Focused context = better, more accurate answers

The AI is explicitly told to ONLY use the provided context. If the answer
isn't in the context, it says so honestly instead of making things up.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from .vectorstore import VectorStore, SearchResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """A source reference for the answer."""
    text: str = Field(description="The relevant text snippet")
    source: str = Field(description="Source filename")
    chunk_index: int = Field(description="Chunk position in document")


class QAResponse(BaseModel):
    """
    A complete answer with sources.

    Attributes:
        answer:  The AI-generated answer text
        sources: List of text chunks that were used to generate the answer
        model:   Which AI model generated the answer
    """
    answer: str = Field(description="The generated answer")
    sources: list[Source] = Field(description="Source chunks used for the answer")
    model: str = Field(description="Model used for generation")


# ---------------------------------------------------------------------------
# System prompt — instructions for the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise document analyst. Your job is to answer
questions based ONLY on the provided context from the user's documents.

Rules:
1. ONLY use information from the provided context chunks
2. If the context does not contain enough information, say so clearly
3. Quote relevant passages when possible
4. Be concise and direct
5. If asked about something not in the context, respond:
   "This information is not found in the provided documents."
6. Always respond in the same language as the user's question

Format your answers in clear, readable prose. Use bullet points for lists."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QAAgent:
    """
    Answers questions about documents using Retrieval-Augmented Generation.

    The pipeline:
      User question → Vector search → Build prompt → LLM → Answer

    Supports a "demo mode" that works without an API key: the vector search
    still runs (finding the most relevant chunks), but instead of sending
    them to the LLM, the agent returns the raw chunks as the answer.

    Usage:
        store = VectorStore()
        store.add_chunks(chunks)

        # With API key:
        agent = QAAgent(api_key="your-api-key", vector_store=store)

        # Without API key (demo mode):
        agent = QAAgent(vector_store=store, demo_mode=True)

        response = agent.ask("What is the main topic?")
        print(response.answer)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        top_k: int = 5,
        demo_mode: bool = False,
    ):
        """
        Initialize the QA agent.

        Args:
            vector_store: The VectorStore containing document chunks
            api_key:      Anthropic API key (reads from env if None)
            model:        LLM model to use for answers
            top_k:        How many chunks to retrieve per question
            demo_mode:    If True, skip API calls and return raw context
        """
        self.demo_mode = demo_mode
        self.model = model
        self.vector_store = vector_store
        self.top_k = top_k

        # Only create the API client if we actually need it
        if not demo_mode:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None

    def ask(self, question: str) -> QAResponse:
        """
        Ask a question about the loaded documents.

        Step by step:
          1. Search the vector store for relevant chunks
          2. Build a prompt with the question + context chunks
          3. Send to the LLM (or format locally in demo mode)
          4. Return the answer with source references

        Args:
            question: The user's question in natural language

        Returns:
            QAResponse with the answer and source references
        """
        # --- Step 1: Retrieve relevant chunks ---
        search_results = self.vector_store.search(question, top_k=self.top_k)

        if not search_results:
            return QAResponse(
                answer="No documents have been loaded yet. Please upload a document first.",
                sources=[],
                model=self.model,
            )

        # --- Step 2: Build the context string ---
        context = self._build_context(search_results)

        # --- Step 3: Generate answer ---
        if self.demo_mode:
            # Demo mode: show the retrieved chunks directly
            answer_text = self._build_demo_answer(question, search_results)
        else:
            # Normal mode: send to LLM API
            user_message = (
                f"Context from the user's documents:\n"
                f"---\n"
                f"{context}\n"
                f"---\n\n"
                f"Question: {question}"
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            answer_text = response.content[0].text

        # --- Step 4: Package sources ---
        sources = [
            Source(
                text=r.text[:200] + "..." if len(r.text) > 200 else r.text,
                source=r.source,
                chunk_index=r.chunk_index,
            )
            for r in search_results
        ]

        return QAResponse(
            answer=answer_text,
            sources=sources,
            model="demo-mode" if self.demo_mode else self.model,
        )

    def _build_demo_answer(self, question: str, results: list[SearchResult]) -> str:
        """
        Build an answer in demo mode by presenting the retrieved chunks.

        In demo mode, we can't use the LLM API to synthesize an answer.
        Instead, we show the user exactly what the RAG retrieval found —
        this demonstrates that the vector search is working correctly.
        """
        lines = [
            "**DEMO MODE** — No API key, showing raw retrieval results.\n",
            "The vector search found these relevant passages for your question:\n",
        ]

        for i, result in enumerate(results, 1):
            # Show similarity score as a percentage (lower distance = higher match)
            similarity = max(0, (1 - result.distance)) * 100
            lines.append(
                f"---\n"
                f"**Match {i}** (relevance: {similarity:.0f}%) "
                f"— *{result.source}, Chunk {result.chunk_index}*\n\n"
                f"> {result.text}\n"
            )

        lines.append(
            "\n---\n"
            "*With an API key, the AI would synthesize these chunks into "
            "a coherent answer to your question.*"
        )

        return "\n".join(lines)

    def _build_context(self, results: list[SearchResult]) -> str:
        """
        Format search results into a context string for the prompt.

        Each chunk is labeled with its source and number so the LLM
        can reference them in its answer.
        """
        parts = []
        for i, result in enumerate(results, 1):
            parts.append(
                f"[Chunk {i} | Source: {result.source}]\n"
                f"{result.text}\n"
            )
        return "\n".join(parts)
