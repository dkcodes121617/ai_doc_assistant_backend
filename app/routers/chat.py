from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
import json
import re

from app.core.security import limiter
from app.models.schemas import ChatRequest
from app.services.embeddings import get_embeddings
from app.services.vector_store import query_chroma
from app.services.llm import call_llm

router = APIRouter(prefix="/chat", tags=["chat"])


def _extract_relevant_excerpt(chunk_text: str, query: str, max_chars: int = 350) -> str:
    """
    Return the full chunk text, clipped nicely, instead of trying to term-match
    which causes mismatched highlights.
    """
    text = chunk_text.strip()
    if len(text) > 400:
        return text[:400] + "..."
    return text


@router.post("")
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    document_ids = body.document_ids
    query = body.query

    # 1. Embed the query
    try:
        query_embedding = get_embeddings([query], task_type="RETRIEVAL_QUERY")[0]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to embed query.")

    # 2. Retrieve top-k chunks
    chunks = query_chroma(query_embedding, document_ids, n_results=5)

    # 3. Build system prompt
    if not chunks:
        system_prompt = (
            "You are a helpful document AI assistant. "
            "No relevant context was found for the user's question. "
            "Politely inform them."
        )
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Chunk [{i}] (Page {chunk['source_page_number']}) ---\n{chunk['text']}\n")

        context_text = "\n".join(context_parts)
        system_prompt = (
            "You are a document-based AI assistant. "
            "Answer the user's question based ONLY on the document chunks provided below.\n"
            "Whenever you use information from a chunk, cite it inline using [1], [2], etc.\n"
            "If the answer is not in the context, say 'I do not have enough information to answer that.'\n\n"
            "CONTEXT:\n" + context_text
        )

    # 4. Build citations — include a query-specific highlighted excerpt
    citations = []
    for i, chunk in enumerate(chunks, 1):
        relevant_excerpt = _extract_relevant_excerpt(chunk["text"], query)
        citations.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["source_page_number"],
            "snippet_text": chunk["text"],          # full chunk (for context)
            "relevant_excerpt": relevant_excerpt,    # query-specific highlighted passage
            "index": i,
        })

    # 5. Stream LLM response then send citations
    async def sse_generator():
        for text_chunk in call_llm(system_prompt, query):
            yield {"data": json.dumps({"type": "text", "content": text_chunk})}

        yield {"data": json.dumps({"type": "citations", "citations": citations})}

    return EventSourceResponse(sse_generator())
