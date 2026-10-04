"""RAG chain: retrieval, prompt building, and streaming generation."""

import logging
from typing import Any, AsyncIterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_ollama import ChatOllama

from app.config import settings
from app.rag import vectorstore

logger = logging.getLogger(__name__)

# NOTE: instructions live in the user message on purpose — llama3.2:1b follows
# them much more reliably there than in a system prompt (verified empirically).
ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context passages below.
If the answer is not in the passages, say "I couldn't find this in the uploaded documents."
Cite the passages you use as [1] or [2].

Context passages:
{context}

Question: {question}"""

EMPTY_KB_ANSWER = (
    "The knowledge base is empty — please upload a document first "
    "(PDF, DOCX, TXT, MD, or CSV) and I'll answer from it."
)

NO_MATCH_ANSWER = (
    "I couldn't find this in the uploaded documents. "
    "Try rephrasing the question or uploading a document that covers this topic."
)


def _llm(temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


def retrieve(question: str) -> list[dict[str, Any]]:
    """Fetch top-k chunks and format them as citation-ready source dicts."""
    results = vectorstore.similarity_search_with_score(question, k=settings.top_k)
    sources = []
    for doc, distance in results:
        # Chroma returns a distance (lower = closer); map it to a 0-100 match score.
        score = max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))
        metadata = doc.metadata or {}
        sources.append(
            {
                "content": doc.page_content,
                "filename": metadata.get("filename", "unknown"),
                "page": metadata.get("page"),
                "doc_id": metadata.get("doc_id", ""),
                "score": round(score * 100),
            }
        )
    return sources


def _build_context(sources: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{i}] {s['content']}" for i, s in enumerate(sources, 1))


def _history_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for msg in history[-settings.max_history_messages :]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


async def stream_answer(
    question: str,
    history: list[dict[str, str]],
    sources: list[dict[str, Any]],
) -> AsyncIterator[str]:
    """Yield answer tokens generated strictly from the retrieved sources."""
    if vectorstore.count_chunks() == 0:
        yield EMPTY_KB_ANSWER
        return

    if not sources:
        yield NO_MATCH_ANSWER
        return

    context = _build_context(sources)
    messages = [
        *_history_messages(history),
        HumanMessage(content=ANSWER_PROMPT.format(context=context, question=question)),
    ]
    async for chunk in _llm().astream(messages):
        if chunk.content:
            yield chunk.content


async def generate_suggestions() -> list[str]:
    """Ask the LLM for example questions based on sampled knowledge-base chunks."""
    passages = vectorstore.sample_chunks(limit=3)
    if not passages:
        return []

    prompt = (
        "Based on the following passages from a knowledge base, write 3 short, "
        "diverse example questions a user could ask about this content. "
        "Return ONLY the questions, one per line, without numbering or quotes.\n\n"
        + "\n---\n".join(passages)
    )
    response = await _llm(temperature=0.7).ainvoke([HumanMessage(content=prompt)])
    questions = [
        line.strip().lstrip("-*0123456789. ").strip()
        for line in response.content.splitlines()
        if line.strip()
    ]
    return questions[:3]
