"""
Text Chunker — Splits documents into overlapping text chunks.

WHY DO WE NEED THIS?
=====================
Imagine you have a 50-page document and someone asks: "What was the Q3 revenue?"
The answer is probably in just 1-2 paragraphs. Instead of sending all 50 pages
to the AI (expensive and slow), we:

  1. Split the document into small pieces ("chunks") of ~500 characters each
  2. Later, we search for the most relevant chunks
  3. Only send those few chunks to the AI

WHAT IS "OVERLAP"?
==================
If we split text at exactly every 500 characters, we might cut a sentence
in half. Overlap means each chunk shares some text with the next one:

  Chunk 1: [============================]
  Chunk 2:                   [============================]
  Chunk 3:                                [============================]
                              ^ overlap ^

This way, no sentence is split without context.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TextChunk(BaseModel):
    """
    A single piece of text from a larger document.

    Attributes:
        text:       The chunk content
        chunk_index: Position in the document (0 = first chunk)
        start_char: Character offset where this chunk starts in the original text
        end_char:   Character offset where this chunk ends
        source:     Filename this chunk came from
    """
    text: str = Field(description="The chunk text content")
    chunk_index: int = Field(description="Position of this chunk (0-based)")
    start_char: int = Field(description="Start character position in original text")
    end_char: int = Field(description="End character position in original text")
    source: str = Field(description="Source filename")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Splits text into overlapping chunks for vector search.

    Parameters:
        chunk_size: Target size of each chunk in characters (default: 500)
        overlap:    How many characters overlap between consecutive chunks (default: 100)

    Example:
        chunker = TextChunker(chunk_size=500, overlap=100)
        chunks = chunker.chunk("Very long text...", source="report.pdf")
        # Returns a list of TextChunk objects
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "unknown") -> list[TextChunk]:
        """
        Split text into overlapping chunks.

        How it works (step by step):
          1. Start at position 0
          2. Take `chunk_size` characters → that's one chunk
          3. Move forward by (chunk_size - overlap) characters
          4. Repeat until we reach the end

        Args:
            text:   The full document text to split
            source: Filename for reference (stored in each chunk)

        Returns:
            List of TextChunk objects
        """
        # If the text is empty, return nothing
        if not text or not text.strip():
            return []

        chunks: list[TextChunk] = []
        # "step" is how far we move forward each time
        # If chunk_size=500 and overlap=100, step=400
        step = self.chunk_size - self.overlap
        position = 0
        index = 0

        while position < len(text):
            # Take a slice of text from current position
            end = min(position + self.chunk_size, len(text))
            chunk_text = text[position:end].strip()

            # Only keep chunks that have actual content
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=index,
                    start_char=position,
                    end_char=end,
                    source=source,
                ))
                index += 1

            # Move forward by one step
            position += step

            # Safety: if we've reached the end, stop
            if end >= len(text):
                break

        return chunks
