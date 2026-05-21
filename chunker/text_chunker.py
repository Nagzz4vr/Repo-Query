from __future__ import annotations

import hashlib
import logging
from enum import Enum
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from pydantic import BaseModel, Field, model_validator

from Chunker.shared_models import Chunk, ChunkMethod, ChunkedDocument
from Ingestion.pdf_ingestor import ExtractionResult

logger = logging.getLogger(__name__)




class ChunkStrategy(str, Enum):
    SEMANTIC  = "semantic"
    RECURSIVE = "recursive"
    TOKEN     = "token"




class ChunkerConfig(BaseModel):

    strategy: ChunkStrategy = Field(
        default=ChunkStrategy.RECURSIVE,
        description="Chunking strategy. Chosen at construction; never changed at runtime.",
    )

    # Semantic
    semantic_breakpoint_type: str = Field(
        default="percentile",
        description="SemanticChunker breakpoint type: percentile | standard_deviation | interquartile",
    )
    semantic_breakpoint_threshold: float = Field(default=95.0, gt=0)
    embeddings_model: str = Field(default="models/embedding-001")

    # Recursive
    recursive_chunk_size: int    = Field(default=1000, gt=0)
    recursive_chunk_overlap: int = Field(default=200,  ge=0)
    recursive_separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    # Token
    token_chunk_size: int    = Field(default=512, gt=0)
    token_chunk_overlap: int = Field(default=50,  ge=0)
    token_encoding_name: str = Field(default="cl100k_base")

    @model_validator(mode="after")
    def validate_overlaps(self) -> ChunkerConfig:
        if self.recursive_chunk_overlap >= self.recursive_chunk_size:
            raise ValueError(
                f"recursive_chunk_overlap ({self.recursive_chunk_overlap}) "
                f"must be < recursive_chunk_size ({self.recursive_chunk_size})"
            )
        if self.token_chunk_overlap >= self.token_chunk_size:
            raise ValueError(
                f"token_chunk_overlap ({self.token_chunk_overlap}) "
                f"must be < token_chunk_size ({self.token_chunk_size})"
            )
        return self


class TextChunker:
    """
    Chunks plain text (PDFs, web pages) using a single, explicit strategy.

    Strategy is fixed at construction time via ChunkerConfig.strategy.
    There is no runtime fallback cascade — if the chosen strategy fails,
    a ChunkingError is raised so the caller can handle it explicitly.
    """

    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        google_api_key: Optional[str] = None,
    ) -> None:
        self.config = config or ChunkerConfig()
        self._strategy_method = self._resolve_strategy_method(google_api_key)



    def chunk_extraction_result(self, result: ExtractionResult) -> ChunkedDocument:
        """
        Chunk a PDF ExtractionResult.

        For SEMANTIC strategy: concatenates all pages first so semantic
        boundaries are not artificially broken at page edges.
        For RECURSIVE / TOKEN: chunks page-by-page to preserve provenance.
        """
        if self.config.strategy == ChunkStrategy.SEMANTIC:
            return self._chunk_document_level(result)
        return self._chunk_page_by_page(result)

    def chunk_text(self, text: str, source_label: str = "unknown") -> ChunkedDocument:
        """Chunk an arbitrary string (e.g. scraped web page)."""
        chunks = self._run_strategy(
            text=text,
            filepath=source_label,
            page_number=None,
            chunk_offset=0,
        )
        return ChunkedDocument(
            filepath=source_label,
            chunk_method_used=ChunkMethod(self.config.strategy.value),
            total_chunks=len(chunks),
            chunks=chunks,
        )


    def _chunk_page_by_page(self, result: ExtractionResult) -> ChunkedDocument:
        all_chunks: list[Chunk] = []
        for page in result.pages:
            text = page.text.text
            if not text.strip():
                continue
            all_chunks.extend(
                self._run_strategy(
                    text=text,
                    filepath=result.filepath,
                    page_number=page.page_number,
                    chunk_offset=len(all_chunks),
                )
            )
        return ChunkedDocument(
            filepath=result.filepath,
            chunk_method_used=ChunkMethod(self.config.strategy.value),
            total_chunks=len(all_chunks),
            chunks=all_chunks,
        )

    def _chunk_document_level(self, result: ExtractionResult) -> ChunkedDocument:
        full_text = "\n".join(
            p.text.text for p in result.pages if p.text.text.strip()
        )
        chunks = self._run_strategy(
            text=full_text,
            filepath=result.filepath,
            page_number=None,   # semantic chunks don't map 1-to-1 to pages
            chunk_offset=0,
        )
        return ChunkedDocument(
            filepath=result.filepath,
            chunk_method_used=ChunkMethod.SEMANTIC,
            total_chunks=len(chunks),
            chunks=chunks,
        )


    def _run_strategy(
        self,
        text: str,
        filepath: str,
        page_number: Optional[int],
        chunk_offset: int,
    ) -> list[Chunk]:
        try:
            docs = self._strategy_method(text)
        except Exception as exc:
            raise ChunkingError(
                f"Chunking failed (strategy={self.config.strategy.value}, "
                f"filepath={filepath}, page={page_number})"
            ) from exc

        return self._to_chunk_models(
            docs=docs,
            filepath=filepath,
            page_number=page_number,
            chunk_offset=chunk_offset,
        )

    def _resolve_strategy_method(self, google_api_key: Optional[str]):
        """
        Build and cache the splitter once at construction.
        Returns a callable: str -> list[Document].
        """
        cfg = self.config

        if cfg.strategy == ChunkStrategy.RECURSIVE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.recursive_chunk_size,
                chunk_overlap=cfg.recursive_chunk_overlap,
                separators=cfg.recursive_separators,
                add_start_index=True,
            )
            return splitter.create_documents

        if cfg.strategy == ChunkStrategy.TOKEN:
            splitter = TokenTextSplitter(
                encoding_name=cfg.token_encoding_name,
                chunk_size=cfg.token_chunk_size,
                chunk_overlap=cfg.token_chunk_overlap,
            )
            return splitter.create_documents

        if cfg.strategy == ChunkStrategy.SEMANTIC:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(
                model=cfg.embeddings_model,
                google_api_key=google_api_key,
            )
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=cfg.semantic_breakpoint_type,
                breakpoint_threshold_amount=cfg.semantic_breakpoint_threshold,
                add_start_index=True,
            )
            return splitter.create_documents

        raise ValueError(f"Unknown strategy: {cfg.strategy}")



    def _to_chunk_models(
        self,
        docs: list[Document],
        filepath: str,
        page_number: Optional[int],
        chunk_offset: int,
    ) -> list[Chunk]:
        result: list[Chunk] = []
        for i, doc in enumerate(docs):
            text = doc.page_content
            if not text.strip():
                continue
            # LangChain provides start_index when add_start_index=True
            # — no post-hoc .find() needed
            char_start: Optional[int] = doc.metadata.get("start_index")

            result.append(Chunk(
                chunk_id=self._make_chunk_id(filepath, text),
                text=text,
                raw_code="",
                method=ChunkMethod(self.config.strategy.value),
                metadata={
                    "page_number": page_number,
                    "char_start": char_start,
                    "chunk_index": chunk_offset + i,
                    "filepath": filepath,
                },
            ))
        return result

    @staticmethod
    def _make_chunk_id(filepath: str, chunk_text: str) -> str:
        """
        Content-addressed ID: stable across reruns, independent of position.
        If the same text appears twice in a document, IDs will collide —
        acceptable trade-off; position-based IDs break on parallel ingestion.
        """
        key = f"{filepath}::{chunk_text}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class ChunkingError(RuntimeError):
    pass