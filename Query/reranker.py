from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class CrossEncoderReranker:
    """Re-ranks retrieved chunks using a cross-encoder model"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5):
        """Re-rank chunks based on query relevance"""
        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Sort by score and return top_k
        ranked = sorted(
            zip(chunks, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [chunk for chunk, _ in ranked[:top_k]]