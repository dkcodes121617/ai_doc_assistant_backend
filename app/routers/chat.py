from fastapi import APIRouter, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
from starlette.concurrency import iterate_in_threadpool
from sse_starlette.sse import EventSourceResponse
import json
import logging

from app.core.security import limiter
from app.models.schemas import ChatRequest, SuggestionRequest
from app.services.embeddings import get_embeddings
from app.services.vector_store import query_chroma, get_collection
from app.services.llm import call_llm, generate_text
from app.core.errors import friendly_error_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Retrieval tuning: pull a wider set, then keep only the clearly-relevant ones.
RETRIEVE_K = 8          # candidates fetched from the vector store
MAX_CONTEXT_CHUNKS = 6  # hard cap on chunks sent to the LLM
MIN_CONTEXT_CHUNKS = 3  # always keep at least this many (if available)
RELATIVE_DISTANCE_CUTOFF = 1.6  # drop chunks farther than best_distance * this


def _filter_relevant(chunks: list[dict]) -> list[dict]:
    """Keep the strongest matches: always the top MIN_CONTEXT_CHUNKS, plus any
    others within a relative distance of the best match, capped at MAX. This is
    metric-agnostic (works whatever Chroma's distance space is) and can only
    trim clearly-worse tail chunks — it never drops the best result."""
    if not chunks:
        return []
    distances = [c.get("distance") for c in chunks]
    if any(d is None for d in distances):
        return chunks[:MAX_CONTEXT_CHUNKS]

    best = distances[0] if distances[0] and distances[0] > 0 else None
    kept = []
    for i, chunk in enumerate(chunks):
        if i < MIN_CONTEXT_CHUNKS:
            kept.append(chunk)
        elif best is not None and chunk["distance"] <= best * RELATIVE_DISTANCE_CUTOFF:
            kept.append(chunk)
        else:
            break
        if len(kept) >= MAX_CONTEXT_CHUNKS:
            break
    return kept

@router.post("/suggestions")
@limiter.limit("10/minute")
async def suggestions(request: Request, body: SuggestionRequest):
    defaults = ["Summarise the key points", "What are the main conclusions?", "List all definitions", "What data or statistics are cited?"]
    try:
        collection = get_collection()
        results = await run_in_threadpool(
            collection.get,
            where={"document_id": {"$in": body.document_ids}},
            limit=4,
        )
    except Exception:
        results = None
        
    if not results or not results["documents"]:
        return {"questions": defaults}
        
    context = "\n".join(results["documents"])
    system_prompt = "You are a helpful AI that generates suggested questions based on document context."
    user_prompt = f"Based on this document snippet, generate exactly 4 short, specific questions (max 8 words each) a user could ask. Return ONLY a valid JSON array of strings containing the questions. Do not include markdown code blocks like ```json.\n\nContext:\n{context}"
    
    try:
        response_text = await run_in_threadpool(generate_text, system_prompt, user_prompt)
        # Clean potential markdown from LLM
        clean_json = response_text.strip().removeprefix("```json").removesuffix("```").strip()
        questions = json.loads(clean_json)
        if isinstance(questions, list) and len(questions) > 0:
            return {"questions": questions[:4]}
    except Exception as e:
        logger.warning("Failed to generate suggestions: %s", e)

    return {"questions": defaults}


@router.post("")
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    document_ids = body.document_ids
    query = body.query

    # 1. Embed the query (blocking network call → run off the event loop)
    try:
        embeddings = await run_in_threadpool(get_embeddings, [query], "RETRIEVAL_QUERY")
        query_embedding = embeddings[0]
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
        raise HTTPException(status_code=500, detail=friendly_error_message(e, context="chat"))

    # 2. Retrieve candidates, then keep only the clearly-relevant ones
    chunks = await run_in_threadpool(query_chroma, query_embedding, document_ids, RETRIEVE_K)
    chunks = _filter_relevant(chunks)

    # 3. Use retrieved chunks directly as separate citations (no page grouping)
    # This allows multiple distinct citations pointing to different specific paragraphs on the same page.
    merged_chunks = []
    # Deduplicate exact text matches just in case
    seen_texts = set()
    for chunk in chunks:
        text = chunk["text"].strip()
        if text not in seen_texts:
            seen_texts.add(text)
            merged_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "source_page_number": chunk["source_page_number"],
                "text": text
            })

    # 4. Build system prompt
    if not merged_chunks:
        system_prompt = (
            "You are a helpful document AI assistant. "
            "No relevant context was found for the user's question. "
            "Politely inform them."
        )
    else:
        context_parts = []
        for i, chunk in enumerate(merged_chunks, 1):
            context_parts.append(f"--- Source [{i}] (Page {chunk['source_page_number']}) ---\n{chunk['text']}\n")

        context_text = "\n".join(context_parts)
        system_prompt = (
            "You are a highly accurate, expert financial and document AI analyst. "
            "Your sole purpose is to answer the user's question based STRICTLY and ONLY on the provided document sources.\n"
            "CRITICAL RULES:\n"
            "1. NO HALLUCINATION: You must not invent, infer, or assume any information, numbers, or facts that are not explicitly stated in the context.\n"
            "2. EXACT NUMBERS: If dealing with financial data, metrics, or statistics, you must quote the exact numbers from the context.\n"
            "3. MANDATORY CITATIONS: Whenever you state a fact, number, or claim from a source, you MUST cite it inline immediately using its bracketed index (e.g., [1] or [1][2]).\n"
            "4. MISSING INFO: If the provided context does not clearly contain the answer, you must truthfully state: 'I do not have enough information to answer that based on the provided documents.' Do not try to guess.\n\n"
            "CONTEXT:\n" + context_text
        )

    # 5. Build citations
    citations = []
    for i, chunk in enumerate(merged_chunks, 1):
        citations.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["source_page_number"],
            "snippet_text": chunk["text"],          # contains all merged paragraphs
            "relevant_excerpt": chunk["text"],      # frontend uses this for display
            "index": i,
        })

    # 5. Stream LLM response then send citations.
    # call_llm is a blocking generator; iterate_in_threadpool pumps it from a
    # worker thread so the event loop stays free for other requests.
    async def sse_generator():
        try:
            async for text_chunk in iterate_in_threadpool(call_llm(system_prompt, query)):
                yield {"data": json.dumps({"type": "text", "content": text_chunk})}
            yield {"data": json.dumps({"type": "citations", "citations": citations})}
        except Exception as e:
            logger.error("Chat streaming failed: %s", e)
            yield {"data": json.dumps({"type": "error", "message": friendly_error_message(e, context="chat")})}

    return EventSourceResponse(sse_generator())
