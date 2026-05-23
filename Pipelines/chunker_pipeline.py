
import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from Chunker.code_chunker import (
    CodeIntelligenceOrchestrator,
    SourceFile,
)
from Chunker.text_chunker import TextChunker
from Chunker.shared_models import Chunk, ChunkedDocument, ChunkMethod
from Pipelines.ingestion_pipeline import IngestedDocument, SourceType

logger = logging.getLogger(__name__)

@dataclass
class ChunkingResult:
    document_id: str
    source_type: SourceType
    chunked_document: ChunkedDocument

@dataclass
class ChunkingFailure:
    document_id: str
    source: str
    reason: str

class ChunkingPipeline:
    def __init__(self,code_orchestrator: CodeIntelligenceOrchestrator,text_chunker: TextChunker,max_concurrency: int = 8,) -> None:
        self._code_orchestrator = code_orchestrator
        self._text_chunker = text_chunker
        self._semaphore = asyncio.Semaphore(max_concurrency)



    async def run(
        self,
        documents: AsyncIterator[IngestedDocument],
    ) -> AsyncIterator[ChunkingResult | ChunkingFailure]:

        async for doc in documents:
            result = await self._chunk_document(doc)
            yield result
    async def run_concurrent(
        self,
        documents: AsyncIterator[IngestedDocument],
    ) -> AsyncIterator[ChunkingResult | ChunkingFailure]:
        pending: set[asyncio.Task] = set()

        async for doc in documents:
            task = asyncio.create_task(self._chunk_document(doc))
            pending.add(task)
            task.add_done_callback(pending.discard)

            done, pending = await asyncio.wait(
                pending,
                timeout=0,  
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                yield t.result()

        # Drain remaining in-flight tasks
        if pending:
            for t in asyncio.as_completed(pending):
                yield await t


    async def _chunk_document(
        self,
        doc: IngestedDocument,
    ) -> ChunkingResult | ChunkingFailure:
        async with self._semaphore:
            try:
                chunked = await self._route(doc)
                chunked = self._inject_document_context(chunked, doc)
                return ChunkingResult(
                    document_id=doc.document_id,
                    source_type=doc.source_type,
                    chunked_document=chunked,
                )
            except Exception:
                logger.warning(
                    "Chunking failed for document %s (%s)",
                    doc.document_id,
                    doc.source,
                    exc_info=True,
                )
                return ChunkingFailure(
                    document_id=doc.document_id,
                    source=doc.source,
                    reason="Unhandled exception during chunking — see logs",
                )

    async def _route(self, doc: IngestedDocument) -> ChunkedDocument:
        """Dispatch to the right chunker based on source_type."""
        if doc.source_type == SourceType.CODE:
            return await self._chunk_code(doc)
        if doc.source_type in (SourceType.PDF, SourceType.WEB):
            return await self._chunk_text(doc)
        raise ValueError(f"No chunker registered for source_type={doc.source_type!r}")


    async def _chunk_code(self, doc: IngestedDocument) -> ChunkedDocument:
        """
        Adapts IngestedDocument → SourceFile and runs the AST orchestrator
        in a thread so it doesn't block the event loop.
        """
        if not doc.path:
            raise ValueError(f"CODE document {doc.document_id} has no path — cannot build SourceFile")

        source_file = SourceFile(
            filepath=doc.path,
            source_code=doc.content,
            language=doc.language or "python",   # explicit default; callers should always set this
        )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._code_orchestrator.process_file,
            source_file,
        )

    async def _chunk_text(self, doc: IngestedDocument) -> ChunkedDocument:
        source_label = doc.path or doc.url or doc.source

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._text_chunker.chunk_text,
            doc.content,
            source_label,
        )


    @staticmethod
    def _inject_document_context(chunked: ChunkedDocument,doc: IngestedDocument,) -> ChunkedDocument:

        document_context = {
            "document_id": doc.document_id,
            "source_type": doc.source_type.value,
            "source": doc.source,
            "title": doc.title,
            "language": doc.language,
            "created_at": doc.created_at.isoformat(),
            **{f"doc_{k}": v for k, v in doc.metadata.items()},
        }

        enriched_chunks: list[Chunk] = []
        for chunk in chunked.chunks:
            enriched_chunks.append(
                chunk.model_copy(
                    update={"metadata": {**document_context, **chunk.metadata}}
                )
            )

        return chunked.model_copy(update={"chunks": enriched_chunks})