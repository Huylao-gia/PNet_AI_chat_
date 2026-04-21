# filepath: backend/services/vector_store.py
from typing import List
from fastapi import Request
from schemas.models import ContextDocument

async def search_context(query: str, request: Request, top_k: int = 3) -> List[ContextDocument]:
    """Tìm kiếm ngữ nghĩa trong VectorDB đã được nạp sẵn trên RAM."""
    embedder = request.app.state.embedder
    collection = request.app.state.collection
    
    # 1. Chuyển đổi câu hỏi thành Vector
    query_vector = embedder.encode([query], show_progress_bar=False).tolist()
    
    # 2. Truy vấn
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    contexts = []
    for i in range(len(documents)):
        # Chuyển đổi khoảng cách Cosine thành % tự tin
        confidence = round((1 - distances[i]) * 100, 1)
        contexts.append(ContextDocument(
            page=str(metadatas[i].get("page", "N/A")),
            content=documents[i],
            confidence_score=confidence
        ))
        
    return contexts
