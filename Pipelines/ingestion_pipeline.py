from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, AsyncIterator, Optional, Type
from uuid import uuid4

from pydantic import BaseModel, Field

from Ingestion.code_ingestor import CodeIngestor
from Ingestion.pdf_ingestor import PdfIngestor
from Ingestion.web_crawler import UniversalScraper

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    CODE = "code"
    PDF = "pdf"
    WEB = "web"


class ContentMode(str, Enum):
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    MIXED = "mixed"


class IngestedDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))

    source_type: SourceType
    content_mode: ContentMode

    source: str
    path: Optional[str] = None
    url: Optional[str] = None

    title: Optional[str] = None
    language: Optional[str] = None

    content: str

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )



class IngestionPipeline:

    def __init__(self, max_concurrent_scrapes: int = 10) -> None:

        self._scraper: Optional[UniversalScraper] = None
        self._max_concurrent_scrapes = max_concurrent_scrapes
        self._semaphore: Optional[asyncio.Semaphore] = None


    async def __aenter__(self) -> "IngestionPipeline":
        self._scraper = UniversalScraper()
        self._semaphore = asyncio.Semaphore(self._max_concurrent_scrapes)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._scraper and hasattr(self._scraper, "close"):
            await self._scraper.close()


    async def ingest_codebase(
        self,
        local_path: Optional[str] = None,
        github_repo: Optional[str] = None,
        github_token: Optional[str] = None,
    ) -> AsyncIterator[IngestedDocument]:
        """
        Yields one IngestedDocument per source file.

        `CodeIngestor.walk()` is blocking (filesystem or GitHub API calls),
        so we collect it entirely in a thread to avoid blocking the event loop.
        Per-file errors are logged and skipped — one bad file never kills the batch.
        """
        ingestor = CodeIngestor(
            local_path=local_path,
            github_repo=github_repo,
            github_token=github_token,
        )


        loop = asyncio.get_running_loop()
        try:
            files = await loop.run_in_executor(None, list, ingestor.walk())
        except Exception:
            logger.exception("Fatal error walking codebase %s / %s", local_path, github_repo)
            return

        for file in files:
            try:
                yield IngestedDocument(
                    source_type=SourceType.CODE,
                    content_mode=ContentMode.CODE,
                    source=file["repo"],
                    path=file["path"],
                    language=file["language"],
                    content=file["source_code"],
                    metadata={
                        "sha": file["sha"],
                        "size": file["size"],
                        "origin": file["source"],
                    },
                )
            except Exception:
                logger.warning("Skipping malformed file entry: %s", file.get("path"), exc_info=True)



    async def ingest_pdf(self, filepath: str) -> IngestedDocument:
        """
        Extracts text from a PDF in a thread (pdfplumber / pymupdf are CPU-bound).
        Stores a page index in metadata instead of the full layout_map dump,
        which can be several MB on large documents.
        """
        loop = asyncio.get_running_loop()

        def _extract() -> IngestedDocument:
            ingestor = PdfIngestor(filepath)
            result = ingestor.extract()

            return IngestedDocument(
                source_type=SourceType.PDF,
                content_mode=ContentMode.TEXT,
                source=result.filepath,
                path=result.filepath,
                title=result.filepath.split("/")[-1],
                content=result.full_text,
                metadata={
                    "total_pages": result.total_pages,
                    "file_size_mb": result.file_size_mb,

                    "layout_map_page_count": len(result.layout_map.pages)
                    if hasattr(result.layout_map, "pages")
                    else None,
                },
            )

        return await loop.run_in_executor(None, _extract)


    async def ingest_webpage(
        self,
        url: str,
        schema: Type[BaseModel],
        instruction: str,
    ) -> IngestedDocument:
        if self._scraper is None or self._semaphore is None:
            raise RuntimeError(
                "IngestionPipeline must be used as an async context manager: "
                "`async with IngestionPipeline() as pipeline`"
            )

        async with self._semaphore:
            result = await self._scraper.scrape_and_extract(
                url=url,
                schema=schema,
                instruction=instruction,
            )

        scraped_at = result.metadata.scraped_at
        if hasattr(scraped_at, "isoformat"):
            scraped_at = scraped_at.isoformat()

        structured_data: str | None = None
        if result.extracted_data is not None:
            import json
            structured_data = json.dumps(result.extracted_data.model_dump(), default=str)

        return IngestedDocument(
            source_type=SourceType.WEB,
            content_mode=ContentMode.MARKDOWN,
            source=url,
            url=url,
            title=result.metadata.title,
            content=result.compressed_markdown or "",
            metadata={
                "scraped_at": scraped_at,
                "word_count": result.metadata.word_count,
                "content_hash": result.metadata.content_hash,
                "structured_data": structured_data,
            },
        )

    async def ingest_webpages(
        self,
        urls: list[str],
        schema: Type[BaseModel],
        instruction: str,
    ) -> AsyncGenerator[IngestedDocument,None]:
        """
        Scrapes many URLs concurrently, bounded by the semaphore.
        Failed URLs are logged and skipped — one 403 doesn't kill the batch.
        Yields results in completion order (not input order) for lower latency.
        """

        async def _scrape_with_url(url: str) -> IngestedDocument:
            # URL closed over here — no external dict lookup needed after completion
            return await self.ingest_webpage(url, schema, instruction)

        tasks = [asyncio.ensure_future(_scrape_with_url(url)) for url in urls]

        for coro in asyncio.as_completed(tasks):
            try:
                yield await coro
            except Exception:
                logger.warning("A webpage scrape failed", exc_info=True)