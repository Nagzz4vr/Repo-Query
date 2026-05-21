from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Any

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class SymbolType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    FILE = "file"


class RetrievalModality(str, Enum):
    CODE = "code"
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"



@dataclass(frozen=True, slots=True)
class SemanticNode:
    symbol_path: str
    name: str
    type: SymbolType

    raw_source: str

    start_line: int
    end_line: int

    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None



@dataclass(frozen=True, slots=True)
class RetrievalChunk:
    chunk_id: str

    modality: RetrievalModality

    text: str

    semantic_node: Optional[SemanticNode] = None

    parent_chunk_id: Optional[str] = None



@dataclass(frozen=True, slots=True)
class EmbeddingProjection:
    chunk_id: str

    embedding_text: str

    retrieval_metadata: dict[str, Any]

    document_text: str



@dataclass(frozen=True, slots=True)
class VectorRecord:
    id: str

    vector: np.ndarray

    metadata: dict[str, Any]

    document: str



class EmbeddingProjector(Protocol):

    def project(
        self,
        chunk: RetrievalChunk,
    ) -> EmbeddingProjection:
        ...



class CodeEmbeddingProjector:

    def project(
        self,
        chunk: RetrievalChunk,
    ) -> EmbeddingProjection:

        node = chunk.semantic_node

        if node is None:
            raise ValueError(
                "CodeEmbeddingProjector requires semantic_node"
            )

        parts: list[str] = []

        parts.append(f"SymbolType: {node.type.value}")

        parts.append(f"Name: {node.name}")

        if node.parent_symbol:
            parts.append(
                f"Parent: {node.parent_symbol}"
            )

        if node.signature:
            parts.append(
                f"Signature: {node.signature}"
            )

        if node.docstring:
            parts.append(
                f"Docstring: {node.docstring}"
            )

        parts.append(chunk.text)

        embedding_text = "\n".join(parts)

        metadata = {
            "symbol_path": node.symbol_path,
            "symbol_type": node.type.value,
            "name": node.name,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "modality": chunk.modality.value,
        }

        return EmbeddingProjection(
            chunk_id=chunk.chunk_id,
            embedding_text=embedding_text,
            retrieval_metadata=metadata,
            document_text=chunk.text,
        )


class TextEmbeddingProjector:

    def project(
        self,
        chunk: RetrievalChunk,
    ) -> EmbeddingProjection:

        metadata = {
            "modality": chunk.modality.value,
        }

        return EmbeddingProjection(
            chunk_id=chunk.chunk_id,
            embedding_text=chunk.text,
            retrieval_metadata=metadata,
            document_text=chunk.text,
        )



class Embedder:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
    ) -> None:

        self.model = SentenceTransformer(model_name)

    def embed(
        self,
        projections: list[EmbeddingProjection],
    ) -> np.ndarray:

        texts = [
            p.embedding_text
            for p in projections
        ]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
        )

        return np.asarray(vectors)



class VectorRecordAssembler:

    @staticmethod
    def assemble(
        projections: list[EmbeddingProjection],
        vectors: np.ndarray,
    ) -> list[VectorRecord]:

        records: list[VectorRecord] = []

        for projection, vector in zip(
            projections,
            vectors,
        ):

            records.append(
                VectorRecord(
                    id=projection.chunk_id,
                    vector=vector,
                    metadata=projection.retrieval_metadata,
                    document=projection.document_text,
                )
            )

        return records




class EmbeddingPipeline:

    def __init__(
        self,
        embedder: Embedder,
    ) -> None:

        self.embedder = embedder

    def run(
        self,
        chunks: list[RetrievalChunk],
    ) -> list[VectorRecord]:

        projections: list[EmbeddingProjection] = []

        for chunk in chunks:

            projector = self._resolve_projector(chunk)

            projection = projector.project(chunk)

            projections.append(projection)

        vectors = self.embedder.embed(projections)

        return VectorRecordAssembler.assemble(
            projections=projections,
            vectors=vectors,
        )

    def _resolve_projector(
        self,
        chunk: RetrievalChunk,
    ) -> EmbeddingProjector:

        if chunk.modality == RetrievalModality.CODE:
            return CodeEmbeddingProjector()

        return TextEmbeddingProjector()