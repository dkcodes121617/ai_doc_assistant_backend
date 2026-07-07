from google.genai import types

from app.core.clients import get_genai_client
from app.core.retry import with_retry


@with_retry
def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Get embeddings for a list of texts.
    Use task_type='RETRIEVAL_QUERY' for search queries,
    'RETRIEVAL_DOCUMENT' for indexing document chunks.
    """
    if not texts:
        return []

    client = get_genai_client()

    response = client.models.embed_content(
        model='models/gemini-embedding-001',
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )

    return [emb.values for emb in response.embeddings]
