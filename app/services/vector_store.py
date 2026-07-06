import chromadb
from app.core.config import settings

def get_chroma_client():
    if not settings.CHROMA_TENANT:
        # Fallback to ephemeral if not configured
        return chromadb.EphemeralClient()
    
    # Connect to Chroma Cloud
    return chromadb.CloudClient(
        tenant=settings.CHROMA_TENANT,
        database=settings.CHROMA_DATABASE,
        api_key=settings.CHROMA_API_KEY
    )

def add_chunks_to_chroma(chunks: list[dict], embeddings: list[list[float]], collection_name: str = "rag_collection"):
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    
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
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def query_chroma(query_embedding: list[float], document_ids: list[str], n_results: int = 5, collection_name: str = "rag_collection") -> list[dict]:
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return []
        
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"document_id": {"$in": document_ids}} if len(document_ids) > 0 else None
    )
    
    chunks = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "document_id": results["metadatas"][0][i]["document_id"],
                "document_filename": results["metadatas"][0][i]["document_filename"],
                "source_page_number": results["metadatas"][0][i]["source_page_number"],
                "text": results["documents"][0][i]
            })
    return chunks
