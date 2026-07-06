from fastapi import APIRouter, UploadFile, File, Request, HTTPException
import uuid
import magic
from app.core.security import limiter
from app.services.extraction import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx
from app.services.chunking import process_document_chunks
from app.services.embeddings import get_embeddings
from app.services.vector_store import add_chunks_to_chroma
from app.models.schemas import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    # Some environments detect DOCX as zip; allow both
    "application/zip",
    "application/octet-stream",
}

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

@router.post("", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB allowed.")

    mime_type = magic.from_buffer(file_bytes, mime=True)
    
    # Determine effective type — DOCX files are ZIP-based so magic may detect as zip
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

    if is_pdf:
        pages_data = extract_text_from_pdf(file_bytes)
    elif is_docx:
        pages_data = extract_text_from_docx(file_bytes)
    else:
        pages_data = extract_text_from_txt(file_bytes)

    if not pages_data:
        raise HTTPException(status_code=400, detail="No text could be extracted from this file.")

    chunks = process_document_chunks(pages_data, document_id, filename)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not generate text chunks from file.")

    texts_to_embed = [c["text"] for c in chunks]
    try:
        embeddings = get_embeddings(texts_to_embed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        add_chunks_to_chroma(chunks, embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        num_chunks=len(chunks),
        num_pages=len(pages_data)
    )
