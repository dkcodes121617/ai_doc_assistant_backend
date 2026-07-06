from fastapi import APIRouter, UploadFile, File, Request, HTTPException
import uuid
import magic
import json
from sse_starlette.sse import EventSourceResponse

from app.core.security import limiter
from app.services.extraction import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx
from app.services.chunking import process_document_chunks
from app.services.embeddings import get_embeddings
from app.services.vector_store import add_chunks_to_chroma

router = APIRouter(prefix="/upload", tags=["upload"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/zip",
    "application/octet-stream",
}

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

@router.post("")
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB allowed.")

    mime_type = magic.from_buffer(file_bytes, mime=True)
    
    filename_lower = (file.filename or "").lower()
    is_docx = (
        mime_type == DOCX_MIME
        or (mime_type in ("application/zip", "application/octet-stream") and filename_lower.endswith(".docx"))
    )
    is_pdf = mime_type == "application/pdf"
    is_txt = mime_type == "text/plain"

    if not (is_pdf or is_txt or is_docx):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type ({mime_type}). Only PDF, DOCX, and TXT are allowed."
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

            pages_data = []
            for item in extractor:
                if item["type"] == "progress":
                    yield {"data": json.dumps(item)}
                elif item["type"] == "result":
                    pages_data = item["data"]

            if not pages_data:
                yield {"data": json.dumps({"type": "error", "message": "Could not extract any text from the document. Please ensure it is not an empty or image-only file without readable text."})}
                return

            yield {"data": json.dumps({"type": "progress", "status": "Chunking document text...", "percent": 50})}
            final_chunks = process_document_chunks(pages_data, document_id, filename)

            texts = [c["text"] for c in final_chunks]
            total_chunks = len(texts)
            embeddings = []
            batch_size = 50
            for i in range(0, total_chunks, batch_size):
                pct = int(50 + (i / total_chunks) * 35)
                yield {"data": json.dumps({"type": "progress", "status": f"Generating embeddings ({min(i + batch_size, total_chunks)}/{total_chunks})...", "percent": pct})}
                batch = texts[i:i + batch_size]
                batch_emb = get_embeddings(batch, task_type="RETRIEVAL_DOCUMENT")
                embeddings.extend(batch_emb)

            yield {"data": json.dumps({"type": "progress", "status": "Saving to vector database...", "percent": 90})}
            add_chunks_to_chroma(final_chunks, embeddings)

            yield {"data": json.dumps({
                "type": "success", 
                "result": {
                    "document_id": document_id,
                    "filename": filename,
                    "num_chunks": len(final_chunks),
                    "num_pages": len(pages_data)
                }
            })}
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "message": str(e)})}

    return EventSourceResponse(event_generator())
