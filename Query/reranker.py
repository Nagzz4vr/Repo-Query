from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Re-rank and return top_k chunks. Score injected as chunk['rerank_score']."""
        if not chunks:
            return []
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores: np.ndarray = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]

    def score_sentences(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score individual sentences — used by ContextCompressor sentence extractor."""
        if not sentences:
            return np.array([])
        pairs = [(query, s) for s in sentences]
        return self.model.predict(pairs)