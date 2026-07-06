from pydantic import BaseModel, Field
from typing import List, Optional

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    num_pages: int

class ChatRequest(BaseModel):
    document_ids: List[str] = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=1000)

class SuggestionRequest(BaseModel):
    document_ids: List[str] = Field(..., min_length=1)

class Citation(BaseModel):
    chunk_id: str
    page_number: int
    snippet_text: str

# This structure won't be sent directly as JSON, but will be sent via SSE
class ChatCompletionChunk(BaseModel):
    content: str
    citations: Optional[List[Citation]] = None
