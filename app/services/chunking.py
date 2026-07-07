import uuid

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """Recursive character splitter with sliding-window overlap.

    1. Recursively split on \n\n, then \n, then ". ", then " " (keeping the
       separators so no content or word boundary is lost) into atomic pieces
       that each fit within chunk_size.
    2. Greedily pack pieces into chunks up to chunk_size. When a chunk fills up,
       seed the next one with the trailing `chunk_overlap` chars of the previous
       chunk so facts spanning a boundary stay retrievable.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split_atoms(txt: str, sep_index: int) -> list[str]:
        if len(txt) <= chunk_size:
            return [txt]
        if sep_index >= len(separators):
            # No separators left: hard slice.
            return [txt[i:i + chunk_size] for i in range(0, len(txt), chunk_size)]

        separator = separators[sep_index]
        parts = txt.split(separator)
        if len(parts) == 1:
            return split_atoms(txt, sep_index + 1)

        atoms: list[str] = []
        for i, part in enumerate(parts):
            # Re-attach the separator (except after the last part) to preserve text.
            piece = part + (separator if i < len(parts) - 1 else "")
            if not piece:
                continue
            if len(piece) > chunk_size:
                atoms.extend(split_atoms(piece, sep_index + 1))
            else:
                atoms.append(piece)
        return atoms

    atoms = split_atoms(text, 0)

    chunks: list[str] = []
    current = ""
    for atom in atoms:
        if current and len(current) + len(atom) > chunk_size:
            chunks.append(current.strip())
            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = overlap + atom
        else:
            current += atom

    if current.strip():
        chunks.append(current.strip())

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
