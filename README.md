# Smart Document Q&A

An AI-powered Retrieval-Augmented Generation (RAG) system that lets you chat with your documents. Upload PDF, DOCX, or TXT files and ask questions — answers are grounded in your document content with source references.

![CI](https://github.com/eugen-goebel/smart-doc-qa/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Tests](https://img.shields.io/badge/Tests-passed-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## How It Works

```
┌─────────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
│  Document    │───▶│  Text    │───▶│  Vector      │───▶│  Stored   │
│  Upload      │    │  Chunker │    │  Store       │    │  Chunks   │
│  (PDF/DOCX)  │    │  (split) │    │  (ChromaDB)  │    │  (embed)  │
└─────────────┘    └──────────┘    └──────────────┘    └─────┬─────┘
                                                             │
┌─────────────┐    ┌──────────┐    ┌──────────────┐          │
│  Answer +   │◀───│  LLM     │◀───│  Relevant    │◀─────────┘
│  Sources    │    │  API     │    │  Chunks      │   (similarity
└─────────────┘    └──────────┘    └──────────────┘    search)
```

### RAG Pipeline

1. **Document Loading** — Reads PDF, DOCX, or TXT files and extracts plain text
2. **Chunking** — Splits text into overlapping ~500-character pieces
3. **Embedding & Storage** — Each chunk is converted to a vector and stored in ChromaDB
4. **Retrieval** — When you ask a question, the most relevant chunks are found via similarity search
5. **Generation** — The LLM answers your question using only the retrieved context

## Quick Start

```bash
# Clone and setup
git clone https://github.com/eugen-goebel/smart-doc-qa.git
cd smart-doc-qa
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your Anthropic API key

# Run the app
streamlit run app.py
```

The app opens in your browser. Upload a document and start asking questions.

### Try with Sample Data

A sample company report is included at `data/sample_company_report.txt`. Upload it in the app and try questions like:

- "What was the company's revenue in 2025?"
- "Who are the main competitors?"
- "What are the strategic priorities for 2026?"
- "Tell me about the BMW case study"

## Architecture

```
smart-doc-qa/
├── app.py                          # Streamlit web interface
├── agents/
│   ├── document_loader.py          # Reads PDF/DOCX/TXT files
│   ├── chunker.py                  # Splits text into overlapping chunks
│   ├── vectorstore.py              # ChromaDB wrapper for similarity search
│   └── qa_agent.py                 # RAG pipeline: retrieve + generate
├── data/
│   └── sample_company_report.txt   # Sample document for testing
├── tests/
│   ├── test_document_loader.py     # 10 tests
│   ├── test_chunker.py             # 15 tests
│   ├── test_vectorstore.py         # 16 tests
│   └── test_qa_agent.py            # 14 tests
├── requirements.txt
└── README.md
```

### Agent Roles

| Agent | Purpose | API Call? |
|-------|---------|-----------|
| **DocumentLoader** | Reads PDF, DOCX, TXT files and extracts text | No |
| **TextChunker** | Splits text into overlapping chunks for search | No |
| **VectorStore** | Stores chunks and finds relevant ones via similarity | No (local embeddings) |
| **QAAgent** | Sends relevant chunks + question to the LLM for answers | Yes (Anthropic API) |

## Key Concepts

### What is RAG?

**Retrieval-Augmented Generation** combines search with AI generation:
- Instead of sending an entire document to the AI (expensive, limited by context window)
- We first **search** for the most relevant parts, then send only those to the AI
- This is faster, cheaper, and produces more accurate answers

### What are Embeddings?

Text is converted into lists of numbers (vectors) that capture meaning. Similar texts have similar vectors. ChromaDB uses the `all-MiniLM-L6-v2` model to generate these embeddings locally — no API key needed.

### What is Chunking?

Documents are split into overlapping pieces (~500 chars each). The overlap ensures no sentence is cut without context at chunk boundaries.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **AI** | Anthropic API | Answer generation from context |
| **Vector DB** | ChromaDB | Embedding storage and similarity search |
| **Embeddings** | all-MiniLM-L6-v2 | Local text-to-vector conversion |
| **Data Models** | Pydantic v2 | Type-safe data validation |
| **Web UI** | Streamlit | Interactive chat interface |
| **PDF Reading** | pypdf | PDF text extraction |
| **DOCX Reading** | python-docx | Word document text extraction |
| **Testing** | pytest | 55+ unit and integration tests |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific agent
pytest tests/test_vectorstore.py -v
```

All tests run without an API key. The QA agent tests use mocked API responses.

## License

MIT
