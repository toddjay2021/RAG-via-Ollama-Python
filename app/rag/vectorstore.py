"""Chroma vector store management plus the document manifest."""

import json
import logging
import random
import threading
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_vectorstore: Chroma | None = None

# Serializes manifest read-modify-write cycles so concurrent uploads
# (e.g. rapid double submissions) cannot overwrite each other's entries.
_manifest_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    """Return the process-wide Chroma instance (lazy init, thread-safe)."""
    global _vectorstore
    with _lock:
        if _vectorstore is None:
            embeddings = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
            )
            _vectorstore = Chroma(
                collection_name="knowledge_base",
                embedding_function=embeddings,
                persist_directory=str(settings.chroma_dir),
            )
        return _vectorstore


# ------------------------------------------------------------------ manifest


def _read_manifest() -> list[dict[str, Any]]:
    if not settings.manifest_path.exists():
        return []
    try:
        return json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read manifest; starting fresh")
        return []


def _write_manifest(docs: list[dict[str, Any]]) -> None:
    settings.manifest_path.write_text(
        json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def list_documents() -> list[dict[str, Any]]:
    """Return all registered documents, most recent first."""
    return sorted(
        _read_manifest(), key=lambda d: d.get("uploaded_at", ""), reverse=True
    )


def register_document(
    doc_id: str, filename: str, chunks: list[Document], file_size: int
) -> None:
    """Embed chunks into Chroma and record the document in the manifest."""
    with _manifest_lock:
        get_vectorstore().add_documents(chunks)
        docs = _read_manifest()
        docs.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": len(chunks),
                "size": file_size,
                "uploaded_at": chunks[0].metadata.get("uploaded_at", "") if chunks else "",
            }
        )
        _write_manifest(docs)


def remove_document(doc_id: str) -> bool:
    """Delete a document's chunks, uploaded file, and manifest entry."""
    with _manifest_lock:
        get_vectorstore().delete(where={"doc_id": doc_id})
        docs = [d for d in _read_manifest() if d.get("doc_id") != doc_id]
        _write_manifest(docs)
    for path in settings.uploads_dir.glob(f"{doc_id}.*"):
        path.unlink(missing_ok=True)
    return True


# ------------------------------------------------------------------- search


def similarity_search_with_score(question: str, k: int) -> list[tuple[Document, float]]:
    """Return (chunk, distance) pairs, closest first."""
    return get_vectorstore().similarity_search_with_score(question, k=k)


def count_chunks() -> int:
    """Total number of chunks currently stored in Chroma."""
    return len(get_vectorstore())


def sample_chunks(limit: int = 3) -> list[str]:
    """Grab a few random chunk texts (used to suggest example questions)."""
    store = get_vectorstore()
    total = len(store)
    if total == 0:
        return []
    offset = random.randint(0, max(0, total - limit))
    result = store.get(limit=limit, offset=offset, include=["documents"])
    return [text for text in result.get("documents", []) if text]
