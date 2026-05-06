# Contributing

Thanks for your interest! This is primarily a personal portfolio project, but contributions are welcome.

## Getting Started

1. Fork the repository and clone your fork.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY` (tests work without one — they mock the client).
5. Run the test suite:
   ```bash
   pytest -v
   ```
6. Try the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Submitting Changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature
   ```
2. Make focused, well-described commits.
3. Make sure the test suite passes locally before pushing.
4. Open a pull request against `main` with a clear description of what you changed and why. Reference any related issues.

## Code Style

- Follow PEP 8 for Python code.
- Add tests for any new behavior — agent tests should mock the Anthropic client.
- Do not commit any vector store data (`.chroma_data/`) or sample documents you do not own.
- Update the README if user-facing behavior changes.
- Keep changes focused — one PR, one concern.
