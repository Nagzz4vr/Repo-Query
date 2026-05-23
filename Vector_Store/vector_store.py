from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

class VectorStoreConfig(BaseModel):
    embedding_dim: int = 1024

    index_type: str = "flat_ip"

    save_dir: Optional[str] = None


class VectorRecord(BaseModel):
    id: str

    vector: list[float]

    metadata: dict

    document: str


class VectorStore:

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
    ) -> None:

        self.config = config or VectorStoreConfig()

        self.index = self._build_index()

        self.records: dict[int, VectorRecord] = {}

        self._next_idx = 0

        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def add(self,records: list[VectorRecord],) -> list[str]:

        if not records:
            return []

        vectors = np.asarray(
            [r.vector for r in records],
            dtype=np.float32,
        )

        if self.config.index_type == "flat_ip":
            faiss.normalize_L2(vectors)

        self.index.add(vectors)

        ids: list[str] = []
        for record in records:
            internal_idx = self._next_idx
            self.records[internal_idx] = record
            self._tokenized_corpus.append(record.document.lower().split())
            self._next_idx += 1
            ids.append(record.id)

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        return ids


    def search(
        self,
        text_query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        rrf_k: int = 60, 
    ) -> list[dict]:

        if self._next_idx == 0:
            return []

        pool_size = min(top_k * 4, self._next_idx)


        #faiss
        query = np.asarray([query_vector], dtype=np.float32)
        if self.config.index_type == "flat_ip":
            faiss.normalize_L2(query)

        dense_scores, dense_indices = self.index.search(query, pool_size)
        dense_results: list[int] = [
            int(idx) for idx in dense_indices[0] if idx != -1
        ]


        sparse_results: list[int] = []
        if self._bm25 is not None:
            tokenized_query = text_query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            sparse_results = np.argsort(bm25_scores)[::-1][:pool_size].tolist()

        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[dict] = []
        for idx, score in fused:
            record = self.records[idx]
            results.append(
                {
                    "id": record.id,
                    "score": float(score),
                    "document": record.document,
                    "metadata": record.metadata,
                }
            )
        return results
    
    
    def save(self, directory: Optional[str] = None) -> Path:
        save_dir = Path(directory or self.config.save_dir or "vector_store")
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))

        with open(save_dir / "records.pkl", "wb") as f:
            pickle.dump(
                {
                    "records": self.records,
                    "next_idx": self._next_idx,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )
        return save_dir


    def load(self, directory: Optional[str] = None) -> None:
        load_dir = Path(directory or self.config.save_dir or "vector_store")

        self.index = faiss.read_index(str(load_dir / "index.faiss"))

        with open(load_dir / "records.pkl", "rb") as f:
            data = pickle.load(f)

        self.records = data["records"]
        self._next_idx = data["next_idx"]
        self._tokenized_corpus = data.get("tokenized_corpus", []) 
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def _build_index(self) -> faiss.Index:
        dim = self.config.embedding_dim
        match self.config.index_type:
            case "flat_ip":
                return faiss.IndexFlatIP(dim)
            case "hnsw":
                index = faiss.IndexHNSWFlat(dim, 32)
                index.hnsw.efConstruction = 200
                return index
            case _:
                return faiss.IndexFlatL2(dim)