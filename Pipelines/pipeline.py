from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel

from Chunker.code_chunker import ASTChunker
from Chunker.shared_models import ChunkedDocument
from Chunker.text_chunker import TextChunker
from Embedder.embedder import Embedder
from Ingestion.code_ingestor import CodeIngestor
from Ingestion.pdf_ingestor import IngestorConfig, PdfIngestor
from Ingestion.web_crawler import UniversalScraper
from Vector_Store.vector_store import VectorStore

logger = logging.getLogger(__name__)

class _WebContent(BaseModel):
    title: str = ""
    body: str = ""
    summary: str = ""

class Pipeline:
    def __init__(self,file_loc: Optional[str] = None,github_repo: Optional[str] = None,web_link: Optional[str] = None,github_token: Optional[str] = None,
                web_schema: Optional[Type[BaseModel]] = None,web_instruction: str = "Extract the main content of the page.",pdf_config: Optional[IngestorConfig] = None,
                embedder: Optional[Embedder] = None,vector_store: Optional[VectorStore] = None,) -> None:
        if not any([file_loc, github_repo, web_link]):
            raise ValueError(
                "Provide at least one source: file_loc, github_repo, or web_link."
            )
        self.file_loc = file_loc
        self.github_repo = github_repo
        self.web_link = web_link
        self.github_token = github_token
        self.web_schema = web_schema or _WebContent
        self.web_instruction = web_instruction
        self.pdf_config = pdf_config or IngestorConfig()

        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

        self._text_chunker = TextChunker()
        self._code_chunker = ASTChunker()

        # track ingested content hashes to skip duplicates within a run
        self._seen_hashes: set[str] = set()

    async def run(self) -> dict[str, int]:
        """
        Full pipeline: ingest → chunk → embed → store.
        Returns a summary dict: {"chunks_added": N, "sources_processed": M}.
        """
        chunked_docs = await self._ingest_and_chunk()

        total_added = 0
        for doc in chunked_docs:
            embedded = self.embedder.embed_document_with_metadata(doc)
            # attach chunk text so the compressor can read it back from the store
            for item in embedded:
                item["metadata"]["text"] = next(
                    (c.text for c in doc.chunks
                    if c.chunk_index == item["metadata"].get("chunk_index")),
                    "",
                )
            ids = self.vector_store.add(embedded)
            total_added += len(ids)
            logger.info("Stored %d chunks from %s", len(ids), doc.filepath)

        return {
            "chunks_added": total_added,
            "sources_processed": len(chunked_docs),
        }
    
    async def _ingest_and_chunk(self) -> list[ChunkedDocument]:
        docs: list[ChunkedDocument] = []

        if self.file_loc:
            docs.extend(self._ingest_local())

        if self.github_repo:                         # ← top-level, not nested
            docs.extend(self._ingest_github())

        if self.web_link:
            web_doc = await self._ingest_web()
            if web_doc:
                docs.append(web_doc)

        return docs
    
    def _ingest_local(self) -> list[ChunkedDocument]:
        path = Path(self.file_loc)
        if not path.exists():
            raise FileNotFoundError(f"file_loc not found: {path}")

        if path.is_file() and path.suffix.lower() == ".pdf":
            return self._ingest_pdf(path)

        # treat anything else as a code directory
        return self._ingest_local_code(path)

    def _ingest_pdf(self, path: Path) -> list[ChunkedDocument]:
        logger.info("Ingesting PDF: %s", path)
        ingestor = PdfIngestor(filepath=str(path))
        extraction = ingestor.extract()

        if self._is_duplicate(extraction.full_text):
            logger.info("Skipping duplicate PDF: %s", path)
            return []

        doc = self._text_chunker.chunk_extraction_result(extraction)
        logger.info("PDF chunked → %d chunks", doc.total_chunks)
        return [doc]

    def _ingest_local_code(self, root: Path) -> list[ChunkedDocument]:
        logger.info("Ingesting local code repo: %s", root)
        ingestor = CodeIngestor(local_path=str(root))
        return self._chunk_code_files(ingestor)


    def _ingest_github(self) -> list[ChunkedDocument]:
        logger.info("Ingesting GitHub repo: %s", self.github_repo)
        ingestor = CodeIngestor(
            github_repo=self.github_repo,
            github_token=self.github_token,
        )
        return self._chunk_code_files(ingestor)



    def _chunk_code_files(self, ingestor: CodeIngestor) -> list[ChunkedDocument]:
        docs: list[ChunkedDocument] = []
        for file_dict in ingestor.walk():
            if self._is_duplicate(file_dict["source_code"]):
                logger.debug("Skipping duplicate file: %s", file_dict["path"])
                continue

            try:
                doc = self._code_chunker.chunk(file_dict)
                docs.append(doc)
            except Exception as exc:
                logger.warning("Chunking failed for %s: %s", file_dict["path"], exc)

        logger.info("Code repo chunked → %d files, total chunks: %d",
                    len(docs), sum(d.total_chunks for d in docs))
        return docs


    async def _ingest_web(self) -> Optional[ChunkedDocument]:
        logger.info("Ingesting web page: %s", self.web_link)
        scraper = UniversalScraper()
        result = await scraper.scrape_and_extract(
            url=self.web_link,
            schema=self.web_schema,
            instruction=self.web_instruction,
            content_type="webpage",
        )

        if not result.metadata.success:
            logger.warning("Web scrape failed for %s: %s",
                            self.web_link, result.metadata.error_message)
            return None

        text = result.compressed_markdown or result.raw_markdown
        if not text.strip():
            logger.warning("No content extracted from %s", self.web_link)
            return None

        if self._is_duplicate(text):
            logger.info("Skipping duplicate web page: %s", self.web_link)
            return None

        doc = self._text_chunker.chunk_text(
            text=text,
            source_label=self.web_link,
        )
        logger.info("Web page chunked → %d chunks", doc.total_chunks)
        return doc

    def _is_duplicate(self, content: str) -> bool:

        h = hashlib.sha256(content.encode()).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)