from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
import json
import re

from app.core.security import limiter
from app.models.schemas import ChatRequest, SuggestionRequest
from app.services.embeddings import get_embeddings
from app.services.vector_store import query_chroma, get_chroma_client
from app.services.llm import call_llm, generate_text

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/suggestions")
@limiter.limit("10/minute")
async def suggestions(request: Request, body: SuggestionRequest):
    defaults = ["Summarise the key points", "What are the main conclusions?", "List all definitions", "What data or statistics are cited?"]
    client = get_chroma_client()
    try:
        collection = client.get_collection(name="rag_collection")
        results = collection.get(
            where={"document_id": {"$in": body.document_ids}},
            limit=4
        )
    except Exception:
        results = None
        
    if not results or not results["documents"]:
        return {"questions": defaults}
        
    context = "\n".join(results["documents"])
    system_prompt = "You are a helpful AI that generates suggested questions based on document context."
    user_prompt = f"Based on this document snippet, generate exactly 4 short, specific questions (max 8 words each) a user could ask. Return ONLY a valid JSON array of strings containing the questions. Do not include markdown code blocks like ```json.\n\nContext:\n{context}"
    
    try:
        response_text = generate_text(system_prompt, user_prompt)
        # Clean potential markdown from LLM
        clean_json = response_text.strip().removeprefix("```json").removesuffix("```").strip()
        questions = json.loads(clean_json)
        if isinstance(questions, list) and len(questions) > 0:
            return {"questions": questions[:4]}
    except Exception as e:
        print("Failed to generate suggestions", e)
        pass
        
    return {"questions": defaults}


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

    # 2. Retrieve top-k chunks (increased to 10 for better recall on complex/financial docs)
    chunks = query_chroma(query_embedding, document_ids, n_results=10)

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

    # 5. Stream LLM response then send citations
    async def sse_generator():
        for text_chunk in call_llm(system_prompt, query):
            yield {"data": json.dumps({"type": "text", "content": text_chunk})}

        yield {"data": json.dumps({"type": "citations", "citations": citations})}

    return EventSourceResponse(sse_generator())
