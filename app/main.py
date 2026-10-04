"""FastAPI application: API routes + static SPA serving."""

import json
import logging
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.rag import chain, vectorstore
from app.rag.loaders import SUPPORTED_EXTENSIONS, load_and_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Knowledge Base Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# -------------------------------------------------------------------- models


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    stream: bool = True


# --------------------------------------------------------------------- health


@app.get("/api/health")
async def health():
    """Check Ollama connectivity and whether the required models are present."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception as exc:
        logger.exception("Health check failed")
        return {"status": "error", "ollama": False, "detail": str(exc)}

    def has(name: str) -> bool:
        return any(m == name or m.startswith(f"{name}:") for m in models)

    return {
        "status": "ok",
        "ollama": True,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "llm_ready": has(settings.llm_model),
        "embedding_ready": has(settings.embedding_model),
    }


# --------------------------------------------------------------------- upload


def _process_upload(file: UploadFile) -> dict:
    """Validate, store, parse, chunk, and embed a single uploaded file."""
    filename = file.filename or "unnamed"
    result: dict = {"filename": filename}
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        result.update(
            status="error",
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )
        return result

    size_limit = settings.max_upload_size_mb * 1024 * 1024
    content = file.file.read()
    if not content:
        result.update(status="error", detail="The file is empty.")
        return result
    if len(content) > size_limit:
        result.update(
            status="error",
            detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
        )
        return result

    doc_id = uuid.uuid4().hex
    dest = settings.uploads_dir / f"{doc_id}{ext}"
    dest.write_bytes(content)

    try:
        chunks = load_and_split(dest, doc_id, filename)
        if not chunks:
            raise ValueError("the file produced no text chunks")
        vectorstore.register_document(doc_id, filename, chunks, len(content))
    except Exception as exc:
        logger.exception("Failed to process upload %s", filename)
        dest.unlink(missing_ok=True)
        result.update(status="error", detail=f"Could not parse the file: {exc}")
        return result

    result.update(status="ok", doc_id=doc_id, chunks=len(chunks))
    return result


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Accept one or more files and add them to the knowledge base."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    results = [await run_in_threadpool(_process_upload, f) for f in files]
    uploaded = sum(1 for r in results if r["status"] == "ok")
    failed = [r for r in results if r["status"] == "error"]
    if uploaded == 0 and failed:
        raise HTTPException(status_code=400, detail=failed[0]["detail"])
    return {"results": results, "uploaded": uploaded, "total_chunks": vectorstore.count_chunks()}


# ----------------------------------------------------------------- documents


@app.get("/api/documents")
async def get_documents():
    docs = await run_in_threadpool(vectorstore.list_documents)
    return {"documents": docs, "total_chunks": await run_in_threadpool(vectorstore.count_chunks)}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    await run_in_threadpool(vectorstore.remove_document, doc_id)
    return {"status": "ok", "total_chunks": vectorstore.count_chunks()}


# ----------------------------------------------------------------------- chat


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Answer a question from the knowledge base.

    Default mode streams over SSE (sources first, then answer tokens).
    With stream=false, returns the complete answer as a single JSON response —
    used as a fallback where SSE is blocked by proxies.
    """
    if not req.stream:
        sources = await run_in_threadpool(chain.retrieve, req.question)
        answer = ""
        async for token in chain.stream_answer(
            req.question,
            [m.model_dump() for m in req.history],
            sources,
        ):
            answer += token
        return {"sources": sources, "answer": answer}

    async def event_generator():
        try:
            sources = await run_in_threadpool(chain.retrieve, req.question)
            yield {
                "event": "sources",
                "data": json.dumps({"sources": sources}),
            }
            collected = []
            async for token in chain.stream_answer(
                req.question,
                [m.model_dump() for m in req.history],
                sources,
            ):
                collected.append(token)
                yield {"event": "token", "data": json.dumps({"content": token})}
            yield {"event": "done", "data": json.dumps({"answer": "".join(collected)})}
        except Exception as exc:
            logger.exception("Chat pipeline failed")
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}

    return EventSourceResponse(event_generator())


# --------------------------------------------------------------- suggestions


@app.get("/api/suggestions")
async def suggestions():
    """Example questions generated from the current knowledge base."""
    questions = await chain.generate_suggestions()
    return {"questions": questions}


# --------------------------------------------------------------------- static


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
