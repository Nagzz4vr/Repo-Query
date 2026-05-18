
from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from Chunker.shared_models import Chunk, ChunkedDocument
from Chunker.chunk_metadata import ChunkMetadata 

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_chunk(self, chunk: Chunk) -> np.ndarray:
        text = self._build_embedding_text(chunk)
        return self.model.encode(text, normalize_embeddings=True)
    
    def embed_document(self, doc: ChunkedDocument) -> list[np.ndarray]:
        texts = [self._build_embedding_text(c) for c in doc.chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return list(vectors)
    
    def embed_document_with_metadata(self, doc: ChunkedDocument) -> list[dict[str, Any]]:
        vectors = self.embed_document(doc)
        return [
            {
                "vector": vec,
                "metadata": ChunkMetadata.from_chunk(chunk).to_vector_store_dict(), # ← used here
            }
            for vec, chunk in zip(vectors, doc.chunks)
        ]
    
    def _build_embedding_text(self, chunk: Chunk) -> str:
        parts: list[str] = []
        if chunk.chunk_type:    
            parts.append(f"Type: {chunk.chunk_type}")
        if chunk.chunk_name:    
            parts.append(f"Name: {chunk.chunk_name}")
        if chunk.parent_symbol: 
            parts.append(f"Parent: {chunk.parent_symbol}")
        if chunk.docstring:     
            parts.append(f"Docstring: {chunk.docstring}")
        parts.append(chunk.text)
        return "\n".join(parts)
    


