from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from Chunker.chunk_metadata import ChunkMetadata

logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseModel):
    embedding_dim: int = Field(default=1024, description="BAAI/bge-m3 output dim")
    index_type: str = Field(default="flat_l2", description="flat_l2 | flat_ip | hnsw")
    save_dir: Optional[str] = Field(default=None, description="Directory to persist the index")

class VectorStore:

    def __init__(self, config: Optional[VectorStoreConfig] = None) -> None:
        self.config = config or VectorStoreConfig()
        self._store: Optional[FAISS] = None
        self._build_empty_store()

    def add(self, results: list[dict[str, Any]]) -> list[str]:

        vectors, documents, ids = [], [], []

        for item in results:
            doc_id = str(uuid.uuid4())
            vectors.append(item["vector"])
            documents.append(
                Document(
                    page_content=item["metadata"].get("chunk_name", ""),
                    metadata=item["metadata"],
                )
            )
            ids.append(doc_id)

            self._store.add_embeddings(
                text_embeddings=list(zip([d.page_content for d in documents], vectors)),
                metadatas=[d.metadata for d in documents],
                ids=ids,
            )

            logger.info("Added %d chunks to vector store.", len(ids))
            return ids
        
    def search(self,query_vector: np.ndarray,top_k: int = 5) -> list[dict[str, Any]]:
        
        docs_and_scores = self._store.similarity_search_by_vector(
            query_vector, k=top_k
        )
        return [
            {
                "score": score,
                "metadata": ChunkMetadata(**doc.metadata),   
            }
            for doc, score in docs_and_scores
        ]
    
    def save(self, directory: Optional[str] = None) -> Path:
        """Persist index + docstore to disk."""
        save_dir = Path(directory or self.config.save_dir or "vector_store_index")
        save_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(save_dir))
        logger.info("Vector store saved to %s", save_dir)
        return save_dir

    def load(self, directory: Optional[str] = None) -> None:
        """Restore a previously saved index from disk."""
        load_dir = Path(directory or self.config.save_dir or "vector_store_index")
        self._store = FAISS.load_local(
            str(load_dir),
            embeddings=_NoOpEmbeddings(),     
            allow_dangerous_deserialization=True,
        )
        logger.info("Vector store loaded from %s", load_dir)

    def _build_empty_store(self) -> None:
        index = self._make_faiss_index()
        self._store = FAISS(
            embedding_function=_NoOpEmbeddings(),
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def _make_faiss_index(self) -> faiss.Index:
        dim = self.config.embedding_dim
        match self.config.index_type:
            case "flat_ip":
                return faiss.IndexFlatIP(dim)          # cosine-friendly (normalise first)
            case "hnsw":
                index = faiss.IndexHNSWFlat(dim, 32)   # 32 = M param
                index.hnsw.efConstruction = 200
                return index
            case _:                                    # default: flat_l2
                return faiss.IndexFlatL2(dim)
            
class _NoOpEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return []