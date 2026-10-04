"""Document loading and chunking for the knowledge base."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

# File types accepted by the demo
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def _build_loader(file_path: Path):
    """Pick the right LangChain loader based on the file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    if ext == ".docx":
        return Docx2txtLoader(str(file_path))
    if ext == ".csv":
        return CSVLoader(str(file_path))
    # .txt and .md
    return TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)


def _clean_metadata(metadata: dict) -> dict:
    """Keep only scalar values (Chroma rejects nested/None metadata)."""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif value is not None:
            cleaned[key] = str(value)
    return cleaned


def load_and_split(file_path: Path, doc_id: str, original_filename: str) -> list[Document]:
    """Load a file, split it into chunks, and tag each chunk with doc metadata.

    Raises ValueError / loader errors propagate to the caller per file.
    """
    loader = _build_loader(file_path)
    pages = loader.load()
    if not pages:
        raise ValueError("the file contains no extractable text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for chunk in chunks:
        chunk.metadata = _clean_metadata(chunk.metadata)
        chunk.metadata.update(
            {
                "doc_id": doc_id,
                "filename": original_filename,
                "uploaded_at": uploaded_at,
            }
        )
    return chunks
