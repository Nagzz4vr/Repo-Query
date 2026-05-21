from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from pydantic import BaseModel, Field




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



    def add(
        self,
        records: list[VectorRecord],
    ) -> list[str]:

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

            self._next_idx += 1

            ids.append(record.id)

        return ids


    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:

        query = np.asarray(
            [query_vector],
            dtype=np.float32,
        )

        if self.config.index_type == "flat_ip":
            faiss.normalize_L2(query)

        scores, indices = self.index.search(
            query,
            top_k,
        )

        results: list[dict] = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            if idx == -1:
                continue

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


    def save(
        self,
        directory: Optional[str] = None,
    ) -> Path:

        save_dir = Path(
            directory
            or self.config.save_dir
            or "vector_store"
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(save_dir / "index.faiss"),
        )

        with open(
            save_dir / "records.pkl",
            "wb",
        ) as f:

            pickle.dump(
                {
                    "records": self.records,
                    "next_idx": self._next_idx,
                },
                f,
            )

        return save_dir


    def load(
        self,
        directory: Optional[str] = None,
    ) -> None:

        load_dir = Path(
            directory
            or self.config.save_dir
            or "vector_store"
        )

        self.index = faiss.read_index(
            str(load_dir / "index.faiss")
        )

        with open(
            load_dir / "records.pkl",
            "rb",
        ) as f:

            data = pickle.load(f)

        self.records = data["records"]

        self._next_idx = data["next_idx"]



    def _build_index(self):

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