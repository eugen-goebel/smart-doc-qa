"""
Smart Document Q&A — Streamlit Chat Interface.

This is the web application that users interact with. It provides:
  - A file upload area (drag & drop PDF, DOCX, or TXT files)
  - A chat interface to ask questions about the uploaded documents
  - Source references showing which parts of the document were used
  - A DEMO MODE that works without an API key (vector search still runs)

To run:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from agents.document_loader import DocumentLoader
from agents.chunker import TextChunker
from agents.vectorstore import VectorStore
from agents.qa_agent import QAAgent

# Load environment variables (for ANTHROPIC_API_KEY)
load_dotenv()


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state — persists data across Streamlit reruns
#
# Streamlit reruns the entire script on every user interaction.
# "Session state" lets us keep data between reruns (like a mini-database
# that lives as long as the browser tab is open).
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize session state variables if they don't exist yet."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar — File upload and document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 Documents")

    # API key handling — demo mode if no key provided
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API Key (optional)",
            type="password",
            help="Get your key at https://console.anthropic.com/",
        )

    # Determine mode
    demo_mode = not bool(api_key)

    if demo_mode:
        st.info(
            "**Demo Mode** — No API key needed!\n\n"
            "Vector search works fully. Instead of AI-generated "
            "answers, you'll see the raw retrieved chunks.\n\n"
            "Add an API key above for AI-generated answers."
        )
    else:
        st.success("API key set — AI answers enabled")

    st.divider()

    # File upload
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Drag & drop PDF, DOCX, TXT, or Markdown files here",
    )

    # "Load sample" button for quick testing
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_company_report.txt")
    if os.path.exists(sample_path) and "sample_company_report.txt" not in st.session_state.uploaded_files:
        if st.button("📋 Load sample document"):
            with st.spinner("Loading sample..."):
                loader = DocumentLoader()
                chunker = TextChunker(chunk_size=500, overlap=100)

                doc = loader.load(sample_path)
                chunks = chunker.chunk(doc.text, source="sample_company_report.txt")
                st.session_state.vector_store.add_chunks(chunks)
                st.session_state.total_chunks += len(chunks)
                st.session_state.uploaded_files.append("sample_company_report.txt")
                st.rerun()

    # Process uploaded files
    if uploaded:
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, overlap=100)

        for file in uploaded:
            # Skip files we already processed
            if file.name in st.session_state.uploaded_files:
                continue

            with st.spinner(f"Processing {file.name}..."):
                # Save uploaded file to a temp location (because our loader reads from disk)
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(file.name)[1],
                ) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                try:
                    # Step 1: Load the document
                    doc = loader.load(tmp_path)

                    # Step 2: Split into chunks
                    chunks = chunker.chunk(doc.text, source=file.name)

                    # Step 3: Store in vector database
                    st.session_state.vector_store.add_chunks(chunks)
                    st.session_state.total_chunks += len(chunks)

                    # Remember this file
                    st.session_state.uploaded_files.append(file.name)

                    st.success(f"{file.name}: {len(chunks)} chunks indexed")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                finally:
                    os.unlink(tmp_path)  # Clean up temp file

    # Show loaded documents
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Loaded Documents")
        for name in st.session_state.uploaded_files:
            st.write(f"✓ {name}")
        st.caption(f"{st.session_state.total_chunks} chunks in vector store")

        # Reset button
        if st.button("🗑️ Clear all documents"):
            st.session_state.vector_store.reset()
            st.session_state.uploaded_files = []
            st.session_state.total_chunks = 0
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Upload a document or load the sample to get started.")


# ---------------------------------------------------------------------------
# Main area — Chat interface
# ---------------------------------------------------------------------------

st.title("Smart Document Q&A")

if demo_mode:
    st.caption(
        "**Demo Mode** — Upload a document, then ask questions. "
        "The vector search finds relevant passages. Add an API key for full AI-generated answers."
    )
else:
    st.caption("Ask questions about your uploaded documents. Answers are based only on the document content.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** (Chunk {src['chunk_index']})")
                    st.caption(src["text"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check prerequisites
    if not st.session_state.uploaded_files:
        st.error("Please upload at least one document first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..." if demo_mode else "Searching documents and generating answer..."):
            agent = QAAgent(
                vector_store=st.session_state.vector_store,
                api_key=api_key if not demo_mode else None,
                demo_mode=demo_mode,
            )
            response = agent.ask(prompt)

        st.markdown(response.answer)

        # Show sources (in demo mode the sources are already in the answer,
        # but we still show them in the expander for consistency)
        if response.sources and not demo_mode:
            with st.expander("📚 Sources", expanded=False):
                for src in response.sources:
                    st.markdown(f"**{src.source}** (Chunk {src.chunk_index})")
                    st.caption(src.text)

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "sources": [s.model_dump() for s in response.sources] if not demo_mode else [],
    })
