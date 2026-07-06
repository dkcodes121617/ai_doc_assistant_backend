import uuid

def chunk_text(text: str, chunk_size=2400, overlap=360) -> list[str]:
    # Approximate 1 token = 4 chars. chunk_size 2400 chars ~ 600 tokens. overlap 360 ~ 15%.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:end].strip())
            break
            
        # Try to find a good break point (newline or period) within the last 150 chars
        break_point = -1
        for sep in ["\n\n", "\n", ". "]:
            sep_idx = text.rfind(sep, start + chunk_size - 150, end)
            if sep_idx != -1:
                break_point = sep_idx + len(sep)
                break
        
        if break_point != -1:
            end = break_point
            
        chunks.append(text[start:end].strip())
        start = end - overlap
        
        if start >= len(text):
            break
            
    return [c for c in chunks if c]

def process_document_chunks(pages_data: list[dict], document_id: str, filename: str) -> list[dict]:
    final_chunks = []
    for page in pages_data:
        text = page["text"]
        if not text:
            continue
        page_num = page["page_number"]
        chunks = chunk_text(text)
        for chunk in chunks:
            final_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "document_filename": filename,
                "source_page_number": page_num,
                "text": chunk
            })
    return final_chunks
