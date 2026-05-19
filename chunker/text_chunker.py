from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from Ingestion.pdf_ingestor import ExtractionResult, PageResult
from Chunker.shared_models import Chunk, ChunkMethod, ChunkedDocument   

logger = logging.getLogger(__name__)

class ChunkerConfig(BaseModel):

    semantic_breakpoint_type: str = Field(
        default="percentile",
        description="SemanticChunker breakpoint type: 'percentile' | 'standard_deviation' | 'interquartile'",
    )
    semantic_breakpoint_threshold: float = Field(
        default=95.0,
        gt=0,
        description="Threshold value passed to SemanticChunker",
    )
    embeddings_model: str = Field(
        default="models/embedding-001",
        description="Gemini embedding model",
    )

    recursive_chunk_size: int = Field(default=1000, gt=0)
    recursive_chunk_overlap: int = Field(default=200, ge=0)
    recursive_separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    token_chunk_size: int = Field(default=512, gt=0)
    token_chunk_overlap: int = Field(default=50, ge=0)
    token_encoding_name: str = Field(
        default="cl100k_base",
        description="tiktoken encoding name",
    )

    add_start_index: bool = Field(
        default=True,
        description="Annotate each chunk with its char offset in the source text",
    )


#chunker for pdf_ingestor
class TextChunker:
    def __init__(self,config: Optional[ChunkerConfig] = None, google_api_key: Optional[str] = None,):
        self.config = config or ChunkerConfig()
        self._google_api_key = google_api_key

    def chunk_extraction_result(self, result: ExtractionResult) -> ChunkedDocument:
        all_chunks: list[Chunk] = []
        method_used: Optional[ChunkMethod] = None

        for page_result in result.pages:
            page_text = page_result.text.text
            if not page_text.strip():
                continue

            chunks, method = self._chunk_text(
                text=page_text,
                page_number=page_result.page_number,
                chunk_offset=len(all_chunks),
            )
            all_chunks.extend(chunks)
            if method_used is None:
                method_used = method

        return ChunkedDocument(
            filepath=result.filepath,
            total_pages=result.total_pages,
            chunk_method_used=method_used or ChunkMethod.TOKEN,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
        )
    
    #text chunker for web pages
    def chunk_text(self,text: str,source_label: str = "unknown",page_number: int = -1) -> ChunkedDocument:
        chunks, method = self._chunk_text(
            text=text,
            page_number=page_number,
            chunk_offset=0,
        )
        return ChunkedDocument(
            filepath=source_label,
            total_pages=1,
            chunk_method_used=method,
            total_chunks=len(chunks),
            chunks=chunks,
        )
    
    def _chunk_text(self,text: str,page_number: int,chunk_offset: int,) -> tuple[list[Chunk], ChunkMethod]:
        for strategy, method in [(self._try_semantic, ChunkMethod.SEMANTIC),(self._try_recursive, ChunkMethod.RECURSIVE),(self._try_token, ChunkMethod.TOKEN)]:
            try:
                raw_chunks = strategy(text)
                if raw_chunks:
                    chunks = self._to_chunk_models(
                        raw_chunks, page_number, chunk_offset, method, text
                    )
                    logger.debug(
                        "page=%d  method=%s  chunks=%d",
                        page_number, method.value, len(chunks),
                    )
                    return chunks, method
            except Exception as exc:
                logger.warning(
                    "Chunking strategy '%s' failed for page %d: %s — trying next.",
                    method.value, page_number, exc,
                )
        logger.error(
            "All chunking strategies failed for page %d. Returning single chunk.",
            page_number,
        )
        fallback = [Chunk(
            chunk_index=chunk_offset,
            page_number=page_number,
            text=text,
            char_start=0,
            method=ChunkMethod.TOKEN,
        )]
        return fallback, ChunkMethod.TOKEN
    
    def _try_semantic(self, text: str) -> list[str]:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embeddings_model,
            google_api_key=self._google_api_key,
        )
    
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=self.config.semantic_breakpoint_type,
            breakpoint_threshold_amount=self.config.semantic_breakpoint_threshold,
            add_start_index=self.config.add_start_index,
        )
    
        docs = splitter.create_documents([text])
    
        return [
            d.page_content
            for d in docs
            if d.page_content.strip()
        ]
    
    def _try_recursive(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.recursive_chunk_size,
            chunk_overlap=self.config.recursive_chunk_overlap,
            separators=self.config.recursive_separators,
            add_start_index=self.config.add_start_index,
        )
        docs = splitter.create_documents([text])
        return [d.page_content for d in docs if d.page_content.strip()]
    
    def _try_token(self, text: str) -> list[str]:
        splitter = TokenTextSplitter(
            encoding_name=self.config.token_encoding_name,
            chunk_size=self.config.token_chunk_size,
            chunk_overlap=self.config.token_chunk_overlap,
        )
        docs = splitter.create_documents([text])
        return [d.page_content for d in docs if d.page_content.strip()]
    
    def _to_chunk_models(self,raw_chunks: list[str],page_number: int,chunk_offset: int,method: ChunkMethod,source_text: str) -> list[Chunk]:
        result: list[Chunk] = []
        search_start = 0

        for i, chunk_text in enumerate(raw_chunks):
            char_start: Optional[int] = None

            if self.config.add_start_index:
                idx = source_text.find(chunk_text, search_start)
                if idx != -1:
                    char_start = idx
                    search_start = idx + len(chunk_text)

            result.append(
                Chunk(
                    chunk_index=chunk_offset + i,
                    page_number=page_number,
                    text=chunk_text,
                    char_start=char_start,
                    method=method,
                )
            )

        return result