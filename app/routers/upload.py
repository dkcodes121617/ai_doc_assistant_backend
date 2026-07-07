from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
from starlette.concurrency import iterate_in_threadpool
import asyncio
import time
import uuid
import json
import logging
import filetype
from sse_starlette.sse import EventSourceResponse

from app.core.security import limiter
from app.services.extraction import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx
from app.services.chunking import process_document_chunks
from app.services.embeddings import get_embeddings
from app.services.vector_store import add_chunks_to_chroma
from app.core.errors import friendly_error_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

@router.post("")
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB allowed.")

    # Detect type from magic bytes (pure-Python, no system libmagic needed).
    kind = filetype.guess(file_bytes)
    mime_type = kind.mime if kind else None

    filename_lower = (file.filename or "").lower()
    is_pdf = mime_type == "application/pdf"
    is_docx = (
        mime_type == DOCX_MIME
        or (mime_type in ("application/zip", "application/octet-stream", None) and filename_lower.endswith(".docx"))
    )
    # filetype cannot sniff plain text (it has no magic-byte signature),
    # so fall back to the .txt extension plus a UTF-8 decodability check.
    is_txt = False
    if not is_pdf and not is_docx and filename_lower.endswith(".txt"):
        try:
            file_bytes.decode("utf-8")
            is_txt = True
        except UnicodeDecodeError:
            is_txt = False

    if not (is_pdf or is_txt or is_docx):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type ({mime_type or 'unknown'}). Only PDF, DOCX, and TXT are allowed."
        )

    filename = file.filename or "unknown_file"
    document_id = str(uuid.uuid4())

    async def event_generator():
        try:
            yield {"data": json.dumps({"type": "progress", "status": "Reading file...", "percent": 0})}
            
            if is_pdf:
                extractor = extract_text_from_pdf(file_bytes)
            elif is_docx:
                extractor = extract_text_from_docx(file_bytes)
            else:
                extractor = extract_text_from_txt(file_bytes)

            # extractor is a blocking generator (pdfplumber / OCR) — pump it
            # from a worker thread so the event loop stays responsive.
            pages_data = []
            async for item in iterate_in_threadpool(extractor):
                if item["type"] == "progress":
                    yield {"data": json.dumps(item)}
                elif item["type"] == "result":
                    pages_data = item["data"]

            if not pages_data:
                yield {"data": json.dumps({"type": "error", "message": "We couldn't read any text from this document. It may be empty, or an image-only/scanned file our OCR couldn't process. Please try a clearer or text-based document."})}
                return

            yield {"data": json.dumps({"type": "progress", "status": "Chunking document text...", "percent": 50})}
            t = time.perf_counter()
            final_chunks = await run_in_threadpool(process_document_chunks, pages_data, document_id, filename)
            logger.info("Chunked '%s' into %d chunks in %.2fs", filename, len(final_chunks), time.perf_counter() - t)

            texts = [c["text"] for c in final_chunks]
            total_chunks = len(texts)
            embeddings = []
            batch_size = 50
            t = time.perf_counter()
            for i in range(0, total_chunks, batch_size):
                pct = int(50 + (i / total_chunks) * 35)
                yield {"data": json.dumps({"type": "progress", "status": f"Generating embeddings ({min(i + batch_size, total_chunks)}/{total_chunks})...", "percent": pct})}
                batch = texts[i:i + batch_size]
                batch_emb = await run_in_threadpool(get_embeddings, batch, "RETRIEVAL_DOCUMENT")
                embeddings.extend(batch_emb)
            logger.info("Embedded %d chunks in %.2fs", total_chunks, time.perf_counter() - t)

            yield {"data": json.dumps({"type": "progress", "status": "Saving to vector database...", "percent": 90})}
            t = time.perf_counter()
            # Guard against a hung/slow Chroma write so the user isn't stuck forever.
            await asyncio.wait_for(
                run_in_threadpool(add_chunks_to_chroma, final_chunks, embeddings),
                timeout=90,
            )
            logger.info("Saved %d chunks to Chroma in %.2fs", len(final_chunks), time.perf_counter() - t)

            yield {"data": json.dumps({
                "type": "success", 
                "result": {
                    "document_id": document_id,
                    "filename": filename,
                    "num_chunks": len(final_chunks),
                    "num_pages": len(pages_data)
                }
            })}
        except asyncio.TimeoutError:
            logger.error("[Upload] Vector DB save timed out for '%s'", filename)
            yield {"data": json.dumps({"type": "error", "message": "Saving to the vector database is taking too long — it may be slow or unreachable right now. Please try again."})}
        except Exception as e:
            logger.error("[Upload] Processing failed: %s", e)
            yield {"data": json.dumps({"type": "error", "message": friendly_error_message(e, context="upload")})}

    return EventSourceResponse(event_generator())
