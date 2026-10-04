# RAG Knowledge Base — Demo

A production-style **Retrieval-Augmented Generation (RAG)** demo built with **Python, LangChain, FastAPI, and Ollama**.
Upload your own documents, build a private knowledge base, and chat with it — every answer is grounded in your files with source citations.

Built to demonstrate full-stack AI integration skills: LLM orchestration, vector search, streaming APIs, and a polished single-page frontend.

## Features

- **Document upload** — drag & drop or file picker; supports PDF, DOCX, TXT, MD, CSV
- **Private knowledge base** — documents are chunked, embedded, and stored in a persistent ChromaDB vector store
- **Grounded answers with citations** — the LLM answers only from retrieved passages; each answer shows collapsible source cards with match scores
- **Streaming responses** — answers stream token-by-token over Server-Sent Events (SSE), with an automatic non-streaming fallback
- **Suggested questions** — AI-generated starter questions based on the uploaded content
- **Document management** — list and delete documents; deletion also removes all associated vectors
- **Polished dark UI** — AI-style theme with gradients, glassmorphism, animations, and toasts
- **Local & private** — everything runs on your machine via Ollama; no data leaves your setup

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS SPA (no build step) |
| Backend | FastAPI + Uvicorn |
| Orchestration | LangChain (loaders, text splitters, chains) |
| LLM | `llama3.2:1b` via Ollama |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | ChromaDB (persistent, local) |

## Project Structure

```
app/
├── main.py              # FastAPI routes: upload, list/delete docs, chat (SSE + JSON), health, suggestions
├── config.py            # pydantic-settings configuration (.env)
├── rag/
│   ├── loaders.py       # File loading + chunking (RecursiveCharacterTextSplitter)
│   ├── vectorstore.py   # ChromaDB singleton + document manifest (thread-safe)
│   └── chain.py         # RAG chain: retrieval, grounded prompting, streaming, suggestions
└── static/
    ├── index.html       # SPA layout (sidebar + chat)
    ├── style.css        # Dark AI-style theme
    └── app.js           # SSE parsing, markdown rendering, uploads, chat logic
```

## Getting Started

### Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running — https://ollama.com
3. Pull the two required models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (optional — defaults work out of the box)
#    Edit .env to change models, chunk size, top_k, upload limits, etc.

# 4. Run the server
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

### Usage

1. Upload a document (PDF/DOCX/TXT/MD/CSV) via the sidebar — it's chunked and indexed automatically
2. Click a suggested question or type your own
3. Watch the answer stream in, then expand **Sources** to see which passages were used
4. Delete documents at any time — their vectors are removed too

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3.2:1b` | Generation model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `4` | Passages retrieved per query |
| `MAX_UPLOAD_SIZE_MB` | `20` | Per-file upload limit |

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Checks Ollama connectivity and model availability |
| `POST` | `/api/upload` | Upload files, chunk & index them |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document and its vectors |
| `POST` | `/api/chat` | Ask a question — SSE stream (`stream: true`) or JSON (`stream: false`) |
| `GET` | `/api/suggestions` | AI-generated example questions from the indexed content |

## How RAG Works Here

1. **Ingest** — uploaded files are parsed, split into overlapping chunks (~800 chars), and embedded with `nomic-embed-text`
2. **Store** — chunks live in a persistent ChromaDB collection; a JSON manifest tracks document metadata
3. **Retrieve** — each question is embedded and the top-k most similar chunks are fetched with similarity scores
4. **Generate** — `llama3.2:1b` answers strictly from the retrieved passages, citing them as `[1]`, `[2]`; if the answer isn't in the documents, it says so instead of guessing
