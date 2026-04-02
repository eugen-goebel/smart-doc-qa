"""
Vector Store — Stores and searches text chunks using ChromaDB.

HOW DOES VECTOR SEARCH WORK?
=============================
Normal search (like Ctrl+F) looks for exact word matches.
Vector search is smarter: it understands MEANING.

Example:
  - Document chunk: "The company's revenue grew by 15% in 2025"
  - User question:  "How much money did the business make?"

  Ctrl+F would find nothing (no matching words).
  Vector search finds it because "revenue grew" and "how much money" have
  similar MEANING.

HOW?
  1. Each text is converted into a list of ~384 numbers called an "embedding"
     Think of it as GPS coordinates, but in 384 dimensions instead of 2
  2. Similar texts have similar numbers (close "coordinates")
  3. ChromaDB stores these and finds the closest matches efficiently

ChromaDB handles the embedding automatically using a built-in model
(no extra API key needed for this part).
"""

import chromadb
from pydantic import BaseModel, Field

from .chunker import TextChunk

DEFAULT_PERSIST_DIR = ".chroma_data"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """
    A single search result from the vector store.

    Attributes:
        text:       The chunk text that matched
        source:     Which file this chunk came from
        chunk_index: Position of this chunk in the original document
        distance:   How "far" this result is from the query (lower = better match)
    """
    text: str = Field(description="The matching chunk text")
    source: str = Field(description="Source filename")
    chunk_index: int = Field(description="Chunk position in document")
    distance: float = Field(description="Similarity distance (lower = more similar)")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Wraps ChromaDB to store document chunks and search them by meaning.

    ChromaDB automatically converts text to embeddings using a small
    built-in model (all-MiniLM-L6-v2). No API key needed for this.

    Usage:
        store = VectorStore()
        store.add_chunks([chunk1, chunk2, chunk3])    # store chunks
        results = store.search("What is the revenue?") # find relevant chunks
    """

    def __init__(self, persist_dir: str | None = None, collection_name: str = "documents"):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to save the database (None = in-memory only)
            collection_name: Name for the ChromaDB collection
        """
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        # get_or_create: if the collection exists, reuse it; otherwise create it
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    @property
    def count(self) -> int:
        """How many chunks are currently stored."""
        return self._collection.count()

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """
        Store text chunks in the vector database.

        Each chunk gets:
          - A unique ID (like "doc_0", "doc_1", ...)
          - The text content (ChromaDB auto-generates the embedding)
          - Metadata (source file, chunk position)

        Args:
            chunks: List of TextChunk objects from the Chunker

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # ChromaDB needs lists of: IDs, documents, and metadata
        existing = self._collection.count()
        ids = [f"doc_{existing + i}" for i in range(len(chunks))]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Find the most relevant chunks for a question.

        How it works:
          1. ChromaDB converts the query into an embedding (same as the chunks)
          2. It compares the query embedding to all stored chunk embeddings
          3. Returns the top_k closest matches

        Args:
            query: The user's question (e.g., "What is the company's revenue?")
            top_k: How many results to return (default: 5)

        Returns:
            List of SearchResult objects, sorted by relevance (best first)
        """
        if self._collection.count() == 0:
            return []

        # Limit top_k to the number of stored chunks
        effective_k = min(top_k, self._collection.count())

        results = self._collection.query(
            query_texts=[query],
            n_results=effective_k,
        )

        # ChromaDB returns nested lists (because you can query multiple things at once)
        # We only query one thing, so we take index [0]
        search_results = []
        for i in range(len(results["documents"][0])):
            search_results.append(SearchResult(
                text=results["documents"][0][i],
                source=results["metadatas"][0][i]["source"],
                chunk_index=results["metadatas"][0][i]["chunk_index"],
                distance=round(results["distances"][0][i], 4),
            ))

        return search_results

    def list_sources(self) -> list[str]:
        """Return a sorted list of unique source filenames in the store."""
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])
        sources = {m["source"] for m in all_meta["metadatas"] if "source" in m}
        return sorted(sources)

    def reset(self):
        """Delete all stored chunks (start fresh)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
