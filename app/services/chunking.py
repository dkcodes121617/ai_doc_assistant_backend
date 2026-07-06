import uuid

def chunk_text(text: str, chunk_size=800, chunk_overlap=150) -> list[str]:
    """Robust recursive character text splitter.
    Splits by \n\n, then \n, then . (sentence), then space.
    """
    separators = ["\n\n", "\n", ". ", " "]
    
    def split_recursively(txt: str, sep_index: int) -> list[str]:
        if len(txt) <= chunk_size:
            return [txt]
            
        if sep_index >= len(separators):
            # Fallback: hard slice
            chunks = []
            for i in range(0, len(txt), chunk_size - chunk_overlap):
                chunks.append(txt[i:i+chunk_size])
            return chunks
            
        separator = separators[sep_index]
        splits = txt.split(separator)
        
        # If this separator doesn't exist in the text, try the next one
        if len(splits) == 1:
            return split_recursively(txt, sep_index + 1)
            
        final_chunks = []
        current_chunk = ""
        
        for i, s in enumerate(splits):
            # Re-add the separator if it's not a newline or if we want to preserve sentences
            part = s + (separator if separator in [". ", " "] and i < len(splits)-1 else "")
            
            if len(current_chunk) + len(part) <= chunk_size:
                # Add a space if we are joining newlines to form a coherent paragraph
                join_char = " " if separator == "\n" and current_chunk else ""
                current_chunk = current_chunk + join_char + part if current_chunk else part
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                # If the single part is STILL larger than chunk_size, recurse on it!
                if len(part) > chunk_size:
                    final_chunks.extend(split_recursively(part, sep_index + 1))
                    current_chunk = ""
                else:
                    current_chunk = part
                    
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return final_chunks

    return split_recursively(text, 0)

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
