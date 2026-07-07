import logging
import threading
import time

import chromadb

from app.core.config import settings
from app.core.retry import with_retry

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_collection"

_lock = threading.Lock()
_client = None
_collection = None


def get_chroma_client():
    """Process-wide singleton Chroma client (cloud, or ephemeral fallback)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                if not settings.CHROMA_TENANT:
                    logger.warning("CHROMA_TENANT not set — using in-memory EphemeralClient (data will not persist).")
                    _client = chromadb.EphemeralClient()
                else:
                    logger.info("Connecting to Chroma Cloud (db=%s)...", settings.CHROMA_DATABASE)
                    t = time.perf_counter()
                    _client = chromadb.CloudClient(
                        tenant=settings.CHROMA_TENANT,
                        database=settings.CHROMA_DATABASE,
                        api_key=settings.CHROMA_API_KEY,
                    )
                    logger.info("Chroma Cloud client ready in %.2fs", time.perf_counter() - t)
    return _client


def get_collection(name: str = COLLECTION_NAME):
    """Cached collection handle (avoids a network round-trip per request)."""
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                logger.info("Opening/creating Chroma collection '%s'...", name)
                t = time.perf_counter()
                # embedding_function=None: we always supply precomputed Gemini
                # embeddings, so Chroma must NOT load its default ONNX model
                # (an ~80MB download that stalls on Render's cold/ephemeral disk).
                _collection = get_chroma_client().get_or_create_collection(
                    name=name,
                    embedding_function=None,
                )
                logger.info("Chroma collection ready in %.2fs", time.perf_counter() - t)
    return _collection


@with_retry
def add_chunks_to_chroma(chunks: list[dict], embeddings: list[list[float]], collection_name: str = COLLECTION_NAME):
    collection = get_collection(collection_name)

    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "document_id": chunk["document_id"],
            "document_filename": chunk["document_filename"],
            "source_page_number": chunk["source_page_number"],
        }
        for chunk in chunks
    ]
    documents = [chunk["text"] for chunk in chunks]

    logger.info("collection.add: writing %d chunks...", len(ids))
    t = time.perf_counter()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    logger.info("collection.add finished in %.2fs", time.perf_counter() - t)


@with_retry
def query_chroma(query_embedding: list[float], document_ids: list[str], n_results: int = 5, collection_name: str = COLLECTION_NAME) -> list[dict]:
    collection = get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"document_id": {"$in": document_ids}} if len(document_ids) > 0 else None,
    )

    chunks = []
    if results and results["ids"] and results["ids"][0]:
        distances = results.get("distances") or [[None] * len(results["ids"][0])]
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "document_id": results["metadatas"][0][i]["document_id"],
                "document_filename": results["metadatas"][0][i]["document_filename"],
                "source_page_number": results["metadatas"][0][i]["source_page_number"],
                "text": results["documents"][0][i],
                "distance": distances[0][i],
            })
    return chunks
