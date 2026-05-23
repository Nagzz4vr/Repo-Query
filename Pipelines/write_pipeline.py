from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import numpy as np
from pydantic import BaseModel

from Pipelines.ingestion_pipeline import *
from Pipelines.chunker_pipeline import ChunkingPipeline, ChunkingResult, ChunkingFailure
from Chunker.code_chunker import CodeIntelligenceOrchestrator, PythonASTParser, ChunkPolicyEngine
from Chunker.text_chunker import TextChunker, ChunkerConfig, ChunkStrategy
from Chunker.shared_models import *
from Embedder.embedder import (
    Embedder,
    EmbeddingPipeline,
    RetrievalChunk,
    RetrievalModality,
    SemanticNode,
    SymbolType,
    VectorRecordAssembler,
)
from Vector_Store.vector_store import VectorStore, VectorStoreConfig, VectorRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class WebSchema(BaseModel):
    summary: Optional[str] = None



class WritePipeline:
    """
    Orchestrates the full write path:
        ingest → chunk → embed → store

    Usage
    -----
    pipeline = WritePipeline()
    await pipeline.run(github_repo="org/repo")
    pipeline.vector_store.save()
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        embedding_dim: int = 1024,
        max_concurrent_scrapes: int = 5,
        max_chunk_concurrency: int = 4,
    ) -> None:
        self.chunking_pipeline = self._build_chunking_pipeline(max_chunk_concurrency)
        self.embedding_pipeline = self._build_embedding_pipeline(embedding_model)
        self.vector_store = VectorStore(
            config=VectorStoreConfig(
                embedding_dim=embedding_dim,
                index_type="flat_ip",
            )
        )
        self._max_concurrent_scrapes = max_concurrent_scrapes



    async def run(
        self,
        github_repo: Optional[str] = None,
        github_token: Optional[str] = None,
        local_path: Optional[str] = None,
        pdf_paths: Optional[list[str]] = None,
        urls: Optional[list[str]] = None,
    ) -> dict:
        """
        Run the full ingest → chunk → embed → store pipeline.
        Returns a summary dict with counts.
        """
        success_docs = 0
        failed_docs = 0
        total_chunks = 0
        total_vectors = 0

        async with IngestionPipeline(
            max_concurrent_scrapes=self._max_concurrent_scrapes
        ) as ingestion:

            documents = self._collect_documents(
                ingestion=ingestion,
                github_repo=github_repo,
                github_token=github_token,
                local_path=local_path,
                pdf_paths=pdf_paths or [],
                urls=urls or [],
            )

            async for result in self.chunking_pipeline.run_concurrent(documents):

                if isinstance(result, ChunkingFailure):
                    failed_docs += 1
                    logger.warning(
                        "FAILED | doc=%s | source=%s | reason=%s",
                        result.document_id,
                        result.source,
                        result.reason,
                    )
                    continue

                success_docs += 1
                chunks = result.chunked_document.chunks
                total_chunks += len(chunks)

                if not chunks:
                    continue

                # Chunk → RetrievalChunk → embed → VectorRecord → store
                retrieval_chunks = self._to_retrieval_chunks(chunks, result.source_type)
                records = self.embedding_pipeline.run(retrieval_chunks)
                vector_records = self._to_vector_store_records(records)
                self.vector_store.add(vector_records)
                total_vectors += len(vector_records)

                logger.info(
                    "SUCCESS | doc=%s | type=%s | chunks=%d | vectors=%d",
                    result.document_id,
                    result.source_type.value,
                    len(chunks),
                    len(vector_records),
                )

        summary = {
            "successful_documents": success_docs,
            "failed_documents": failed_docs,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors,
        }

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE: %s", summary)
        return summary



    @staticmethod
    async def _collect_documents(
        ingestion: IngestionPipeline,
        github_repo: Optional[str],
        github_token: Optional[str],
        local_path: Optional[str],
        pdf_paths: list[str],
        urls: list[str],
    ) -> AsyncIterator[IngestedDocument]:

        if github_repo or local_path:
            async for doc in ingestion.ingest_codebase(
                github_repo=github_repo,
                github_token=github_token,
                local_path=local_path,
            ):
                yield doc

        for filepath in pdf_paths:
            try:
                yield await ingestion.ingest_pdf(filepath=filepath)
            except Exception:
                logger.warning("PDF ingestion failed: %s", filepath, exc_info=True)

        if urls:
            async for doc in ingestion.ingest_webpages(
                urls=urls,
                schema=WebSchema,
                instruction="Extract the main educational content. Ignore navigation and footers.",
            ):
                yield doc



    @staticmethod
    def _to_retrieval_chunks(
        chunks: list[Chunk],
        source_type: SourceType,
    ) -> list[RetrievalChunk]:
        """Adapt Chunker.Chunk → Embedder.RetrievalChunk."""

        modality_map = {
            SourceType.CODE: RetrievalModality.CODE,
            SourceType.PDF: RetrievalModality.PDF,
            SourceType.WEB: RetrievalModality.MARKDOWN,
        }
        modality = modality_map.get(source_type, RetrievalModality.TEXT)

        result = []
        for chunk in chunks:
            meta = chunk.metadata

            semantic_node: Optional[SemanticNode] = None
            if modality == RetrievalModality.CODE and meta.get("symbol_path"):
                semantic_node = SemanticNode(
                    symbol_path=meta["symbol_path"],
                    name=meta.get("name", ""),
                    type=SymbolType(meta.get("type", "function")),
                    raw_source=chunk.raw_code or chunk.text,
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    signature=meta.get("signature"),
                    docstring=meta.get("docstring"),
                    parent_symbol=meta.get("parent_path"),
                )

            result.append(RetrievalChunk(
                chunk_id=chunk.chunk_id,
                modality=modality,
                text=chunk.text,
                semantic_node=semantic_node,
                parent_chunk_id=chunk.parent_chunk_id,
            ))

        return result

    @staticmethod
    def _to_vector_store_records(
        records: list,   # list[Embedder.VectorRecord]
    ) -> list[VectorRecord]:
        """Adapt Embedder.VectorRecord → VectorStore.VectorRecord."""
        return [
            VectorRecord(
                id=r.id,
                vector=r.vector.tolist(),
                metadata=r.metadata,
                document=r.document,
            )
            for r in records
        ]



    @staticmethod
    def _build_chunking_pipeline(max_concurrency: int) -> ChunkingPipeline:
        return ChunkingPipeline(
            code_orchestrator=CodeIntelligenceOrchestrator(
                parser_registry={"python": PythonASTParser()},
                policy_engine=ChunkPolicyEngine(min_lines=4, merge_small_methods=True),
            ),
            text_chunker=TextChunker(
                config=ChunkerConfig(
                    strategy=ChunkStrategy.RECURSIVE,
                    recursive_chunk_size=1200,
                    recursive_chunk_overlap=200,
                )
            ),
            max_concurrency=max_concurrency,
        )

    @staticmethod
    def _build_embedding_pipeline(model_name: str) -> EmbeddingPipeline:
        return EmbeddingPipeline(embedder=Embedder(model_name=model_name))