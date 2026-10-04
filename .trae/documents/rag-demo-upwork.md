# RAG Demo for Upwork — Implementation Plan

## Context

Build a customer-facing RAG (Retrieval-Augmented Generation) demo to showcase AI integration skills on Upwork. Customers upload their own files (PDF/DOCX/TXT/MD/CSV) to build a knowledge base, then chat with it. Answers stream in real-time and cite the exact source chunks used — making the RAG mechanism tangible and credible.

**Confirmed decisions:**
- Vanilla SPA (HTML/CSS/JS) served by FastAPI — no build step, single `uvicorn` command to run
- Dark AI-style theme (gradient accents, glow effects, ChatGPT-like feel)
- Features: source citations, streaming responses, suggested questions
- Stack: Python + LangChain 1.x + FastAPI + Ollama (`llama3.2:1b` for generation, `nomic-embed-text` for embeddings — both already pulled in Ollama)

**Environment (verified):**
- venv at `.venv` with langchain 1.4.3, langchain-community 0.4.2, langchain-ollama 1.1.0, chromadb 1.5.9, fastapi 0.142.2, uvicorn, sse-starlette, python-dotenv — all installed
- Ollama running at localhost:11434 with llama3.2:1b + nomic-embed-text
- **Missing packages to install:** `pypdf`, `docx2txt`, `python-multipart`

## Architecture

```
app/
├── __init__.py
├── main.py            # FastAPI app: routes + static serving + CORS
├── config.py          # pydantic-settings: models, paths, RAG params
├── rag/
│   ├── __init__.py
│   ├── loaders.py     # File → Documents (by extension), chunking
│   ├── vectorstore.py # Chroma persistent store: add/delete/list/search
│   └── chain.py       # Retrieval + prompt + ChatOllama streaming generator
└── static/
    ├── index.html     # SPA layout: sidebar + chat
    ├── style.css      # Dark AI theme
    └── app.js         # Upload, SSE chat, suggestions, doc management

data/                  # created at runtime (gitignored)
├── uploads/           # original uploaded files
└── chroma_db/         # persisted vector store
manifest.json          # doc registry: doc_id, filename, chunks, timestamp
```

## Backend

### config.py
pydantic-settings reading `.env` (add `EMBEDDING_MODEL=nomic-embed-text`):
`OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `TOP_K=4`, data dir paths.

### rag/loaders.py
- Map extension → loader: `.pdf`→PyPDFLoader, `.docx`→Docx2txtLoader, `.txt/.md`→TextLoader, `.csv`→CSVLoader (imports verified: `from langchain_community.document_loaders import ...`)
- `load_and_split(file_path)`: load → `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)` (from `langchain_text_splitters`)
- Tag each chunk's metadata: `doc_id`, `source` (filename), `page`, `uploaded_at`

### rag/vectorstore.py
- Singleton `Chroma` from `langchain_community.vectorstores` with `persist_directory="data/chroma_db"`, `collection_name="knowledge_base"`, `embedding_function=OllamaEmbeddings(model="nomic-embed-text", base_url=...)`
- `add_documents(docs)`, `delete_document(doc_id)` (Chroma `where={"doc_id": ...}`), `as_retriever(k=TOP_K)` with score fetching via `similarity_search_with_score`
- Manifest (`data/manifest.json`): list/delete doc metadata (id, name, size, chunks, uploaded_at)

### rag/chain.py
- `retrieve(question)` → top-k chunks + scores (for citations)
- Prompt: strict RAG template — answer ONLY from provided context, admit when context lacks the answer, cite sources. Include short chat history (last 4 messages) for follow-ups.
- `stream_answer(question, history)` → async generator over `ChatOllama(model=llama3.2:1b, temperature=0.1).astream(...)` tokens

### main.py — API endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/upload` | Multipart upload (multi-file) → parse, chunk, embed, store; returns per-file status + chunk count |
| GET | `/api/documents` | List KB documents from manifest |
| DELETE | `/api/documents/{doc_id}` | Remove chunks from Chroma + file + manifest entry |
| POST | `/api/chat` | SSE stream: `sources` event → `token` events → `done` (sse-starlette `EventSourceResponse`) |
| GET | `/api/suggestions` | Sample 3 chunks → llama3.2 generates 3 example questions |
| GET | `/api/health` | Ollama + models availability check |

SSE protocol: `event: sources` (doc name, page, snippet, score) → `event: token` → `event: done` | `event: error`.

## Frontend (dark AI-style SPA)

- **Layout**: left sidebar (KB management) + main chat panel
- **Sidebar**: gradient logo mark + title, drag-&-drop upload zone with progress, document cards (type icon, name, chunk count, delete), KB stats footer (X docs · Y chunks)
- **Chat**: model badge header (llama3.2:1b · via Ollama), welcome hero with suggested-question chips (from `/api/suggestions`, clickable), message bubbles with streaming text + blinking cursor, collapsible "Sources" cards under each answer (doc, page, score bar, snippet), auto-scroll, error toasts
- **Style**: dark bg (#0a0e14 family), purple→blue gradient accents, glassmorphism panels, subtle glow, message fade-in animations, Inter font (Google Fonts with system fallback)
- **JS**: vanilla, no dependencies; small safe markdown renderer (bold/code/lists); SSE via `fetch` + `ReadableStream` parsing (POST required, so no native EventSource)
- All UI text in English (Upwork audience)

## Steps

1. Install missing deps: `pip install pypdf docx2txt python-multipart`
2. Create `app/config.py`, `app/rag/` modules (loaders → vectorstore → chain)
3. Create `app/main.py` with all API routes + static mounting
4. Add `EMBEDDING_MODEL=nomic-embed-text` to `.env`, update `requirements.txt`, extend `.gitignore` (data/)
5. Build frontend: `index.html` → `style.css` → `app.js`
6. End-to-end verification (below)

## Verification

1. Start server: `uvicorn app.main:app --reload` from project root
2. `GET /api/health` → both models reachable
3. Create a sample test file (e.g. a small PDF/TXT with distinct facts), upload via UI drag-drop → appears in sidebar with chunk count
4. Ask a question whose answer is in the file → verify: sources event arrives first, tokens stream, answer is grounded in the file content
5. Ask something unrelated → model should say the context doesn't contain the answer
6. Delete document → chunks removed, follow-up question no longer cites it
7. Browser check of UI: layout, streaming animation, source cards, upload interaction
