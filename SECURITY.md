# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately by emailing **eugen-goebel@hotmail.de**.

Please do not file public GitHub issues for security vulnerabilities, as this could expose users to risk before a fix is available.

## Response Time

I aim to acknowledge reports within 7 days and provide an initial assessment within 14 days.

## Supported Versions

This is a portfolio project; only the latest commit on `main` is supported.

## API Key Handling

This project uses the `ANTHROPIC_API_KEY` environment variable. Never commit your API key — use the provided `.env.example` as a template and keep your real `.env` file out of version control (it is gitignored).

## Document Privacy

When using this RAG system, uploaded documents are processed locally and stored in the local ChromaDB vector store. Only the chunks selected by retrieval are sent to the Claude API at query time. Do not upload sensitive documents to public deployments without reviewing the data flow.
