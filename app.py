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

# Force pure-Python protobuf implementation. The C++ extension's
# generated _pb2.py descriptors (in chromadb -> opentelemetry-proto)
# are incompatible with protobuf 5.x on Python 3.13+, which is the
# default on Streamlit Community Cloud. Must be set before any
# protobuf-using import (streamlit, chromadb).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tempfile

import streamlit as st
from dotenv import load_dotenv

from agents.chunker import TextChunker
from agents.document_loader import DocumentLoader
from agents.qa_agent import QAAgent
from agents.vectorstore import DEFAULT_PERSIST_DIR, VectorStore

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


SAMPLE_DOC_NAME = "sample_company_report.txt"


def init_session_state():
    """Initialize session state variables if they don't exist yet."""
    if "vector_store" not in st.session_state:
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
        st.session_state.vector_store = VectorStore(persist_dir=persist_dir)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        # Restore previously indexed files from persistent store
        st.session_state.uploaded_files = st.session_state.vector_store.list_sources()
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = st.session_state.vector_store.count


def load_sample_document() -> bool:
    """Index the bundled sample document. Returns True if it was loaded."""
    sample_path = os.path.join(os.path.dirname(__file__), "data", SAMPLE_DOC_NAME)
    if not os.path.exists(sample_path):
        return False

    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=500, overlap=100)

    doc = loader.load(sample_path)
    chunks = chunker.chunk(doc.text, source=SAMPLE_DOC_NAME)
    st.session_state.vector_store.add_chunks(chunks)
    st.session_state.total_chunks += len(chunks)
    st.session_state.uploaded_files.append(SAMPLE_DOC_NAME)
    return True


def autoload_sample_once():
    """Preload the sample document so the demo works on first visit.

    Runs at most once per session, and only while the store is empty, so
    clearing all documents is not immediately undone by a reload.
    """
    if st.session_state.get("sample_autoloaded"):
        return
    st.session_state.sample_autoloaded = True
    if not st.session_state.uploaded_files:
        load_sample_document()


init_session_state()
autoload_sample_once()


# ---------------------------------------------------------------------------
# Sidebar — File upload and document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 Documents")

    # API key handling — checks env, then Streamlit secrets, then user input.
    # Demo mode if no key provided.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except (FileNotFoundError, AttributeError):
            api_key = ""
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
    sample_path = os.path.join(os.path.dirname(__file__), "data", SAMPLE_DOC_NAME)
    if os.path.exists(sample_path) and SAMPLE_DOC_NAME not in st.session_state.uploaded_files:
        if st.button("📋 Load sample document"):
            with st.spinner("Loading sample..."):
                load_sample_document()
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
        "**Demo Mode** — A sample company report is preloaded, just ask a question. "
        "Upload your own documents anytime. Add an API key for full AI-generated answers."
    )
else:
    st.caption(
        "Ask questions about your uploaded documents. Answers are based only on the document content."
    )

# A short explainer so a first-time visitor understands what RAG buys them,
# even after a single question. Shown in demo mode, stays visible across turns.
if demo_mode:
    with st.expander("ℹ️ How this works, and why it is useful"):
        st.markdown(
            "This app answers with **retrieval-augmented generation (RAG)**:\n\n"
            "1. Each document is split into small **chunks** that are turned into "
            "numeric **embeddings** and stored in a vector database.\n"
            "2. Your question is embedded the same way, so the app can retrieve the "
            "passages closest **in meaning** — not just keyword matches.\n"
            "3. With an API key, Claude writes a grounded answer and cites its sources. "
            "In demo mode you see the retrieved passages directly.\n\n"
            "The payoff grows with document size: instead of reading a long report, you "
            "ask and it retrieves the relevant passages. The preloaded file is a full "
            "multi-section annual report, so ask about any part of it — the financials, "
            "the customers, the team, the risks, or the 2026 plan."
        )

# Example questions — shown on first visit so the demo is instantly explorable.
EXAMPLE_QUESTIONS = [
    "What was the company's revenue in 2025?",
    "Who are the key executives?",
    "How did the team grow in 2025?",
]
if demo_mode and st.session_state.uploaded_files and not st.session_state.messages:
    st.caption("Try one of these:")
    for col, question in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS, strict=True):
        if col.button(question, use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

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

# Chat input — a question can come from the box or from an example button
typed = st.chat_input("Ask a question about your documents...")
prompt = typed or st.session_state.pop("pending_question", None)
if prompt:
    # Check prerequisites
    if not st.session_state.uploaded_files:
        st.error("Please upload a document first, or click 'Load sample document' in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner(
            "Searching documents..."
            if demo_mode
            else "Searching documents and generating answer..."
        ):
            agent = QAAgent(
                vector_store=st.session_state.vector_store,
                api_key=api_key if not demo_mode else None,
                demo_mode=demo_mode,
            )
            # Forward the chat history (excluding the new user message we
            # just appended) so the model can resolve follow-up questions.
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m.get("role") in ("user", "assistant")
            ]
            response = agent.ask(prompt, conversation_history=history)

        st.markdown(response.answer)

        # Show sources (in demo mode the sources are already in the answer,
        # but we still show them in the expander for consistency)
        if response.sources and not demo_mode:
            with st.expander("📚 Sources", expanded=False):
                for src in response.sources:
                    st.markdown(f"**{src.source}** (Chunk {src.chunk_index})")
                    st.caption(src.text)

    # Save to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": [s.model_dump() for s in response.sources] if not demo_mode else [],
        }
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Built by Eugen Goebel &middot; "
    "<a href='https://github.com/eugen-goebel' target='_blank'>GitHub</a> &middot; "
    "<a href='https://www.linkedin.com/in/eugen-goebel/' target='_blank'>LinkedIn</a>"
    "</div>",
    unsafe_allow_html=True,
)
