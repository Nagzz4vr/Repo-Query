from typing import Optional
from pydantic import BaseModel, field_serializer
from Chunker.shared_models import Chunk, ChunkMethod


class ChunkMetadata(BaseModel):
    chunk_index:   int
    page_number:   int
    method:        ChunkMethod
    chunk_type:    Optional[str] = None
    chunk_name:    Optional[str] = None
    parent_symbol: Optional[str] = None
    start_line:    Optional[int] = None
    end_line:      Optional[int] = None
    path:          Optional[str] = None
    language:      Optional[str] = None

    @field_serializer("method")
    def serialize_method(self, value: ChunkMethod) -> str:
        return value.value

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkMetadata":
        return cls(
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            method=chunk.method,
            chunk_type=chunk.chunk_type,
            chunk_name=chunk.chunk_name,
            parent_symbol=chunk.parent_symbol,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            path=chunk.path,
            language=chunk.language,
        )

    def to_vector_store_dict(self) -> dict:
        """Most vector stores (Pinecone, Qdrant, Chroma) expect flat dicts with no None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}