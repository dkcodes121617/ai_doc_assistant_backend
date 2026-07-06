from google import genai
from google.genai import types
from app.core.config import settings

def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Get embeddings for a list of texts.
    Use task_type='RETRIEVAL_QUERY' for search queries,
    'RETRIEVAL_DOCUMENT' for indexing document chunks.
    """
    if not texts:
        return []
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    response = client.models.embed_content(
        model='models/gemini-embedding-001',
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )
    
    embeddings = []
    for emb in response.embeddings:
        embeddings.append(emb.values)
    return embeddings
