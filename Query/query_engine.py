from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

import numpy as np
from pydantic import BaseModel, Field

from Embedder.embedder import Embedder
from LLM.llm_client import LiteLLMClient
from Memory.context_compressor import ContextCompressor, CompressionConfig
from Memory.memory import ConversationMemory, MemoryConfig
from Query.query_rewriter import QueryRewriter, RewriteConfig
from Query.reranker import CrossEncoderReranker
from Vector_Store.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Add these to QueryConfig
class QueryConfig(BaseModel):
    top_k_retrieval: int = Field(default=10)
    top_k_final: int = Field(default=5)

    system_prompt_code: str = Field(default=(
        "You are an expert code assistant. You are given code snippets from a codebase. "
        "Answer questions directly and confidently based on the code. "
        "Reference class names, function names, and file paths in your answer. "
        "Never say 'without more context' — reason from what you can see in the code."
    ))
    system_prompt_pdf: str = Field(default=(
        "You are an expert research assistant. You are given excerpts from a PDF document. "
        "Answer questions based strictly on the provided text. "
        "Quote or paraphrase relevant passages. Cite page numbers when available. "
        "If the answer is not in the excerpts, say so briefly."
    ))
    system_prompt_web: str = Field(default=(
        "You are a helpful assistant. You are given content scraped from web pages. "
        "Summarize and answer based on the provided content. "
        "Mention the source URL when relevant. Be concise and direct."
    ))
    system_prompt_default: str = Field(default=(
        "You are a helpful assistant. Answer questions using the provided context. "
        "Be direct and cite sources where possible."
    ))

class QueryResult(BaseModel):
    question: str
    rewritten_queries: list[str]
    answer: str
    sources: list[dict[str, Any]]

class QueryEngine:
    def __init__(
        self,
        client: LiteLLMClient,
        embedder: Embedder,
        vector_store: VectorStore,
        query_config: Optional[QueryConfig] = None,
        rewrite_config: Optional[RewriteConfig] = None,
        compression_config: Optional[CompressionConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.client = client
        self.vector_store = vector_store
        self.config = query_config or QueryConfig()

        self.memory = ConversationMemory(memory_config)

        self.rewriter = QueryRewriter(
            client=client,
            embedder=embedder,
            config=rewrite_config,
        )

        reranker = CrossEncoderReranker(model_name=reranker_model)
        self.compressor = ContextCompressor(
            reranker=reranker,
            config=compression_config,
        )
    
    async def query(self, question: str) -> QueryResult:
        """Run the full pipeline and return a structured QueryResult."""
        chunks, queries = await self._run_pipeline(question)
        answer = await self._synthesize(queries[0], chunks)

        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", answer)

        return QueryResult(
            question=question,
            rewritten_queries=queries,
            answer=answer,
            sources=self._extract_sources(chunks),
        )

    async def stream_query(self, question: str) -> AsyncIterator[str]:
        """
        Streaming variant — yields answer tokens as they arrive.
        Memory is updated once the full stream completes.
        """
        chunks, queries = await self._run_pipeline(question)
        context = self._build_context_string(chunks)
        history = self.memory.format_as_messages()
        system_prompt = self._pick_system_prompt(chunks)

        messages = [
            *history,
            {"role": "user", "content": self._user_prompt(queries[0], context)},
        ]

        collected: list[str] = []
        async for token in self.client.stream(
            system_prompt=system_prompt,
            messages=messages,
        ):
            collected.append(token)
            yield token

        full_answer = "".join(collected)
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", full_answer)

    def reset_memory(self) -> None:
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    async def _run_pipeline(
        self,
        question: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        await self.memory.evict_if_needed(self.client)

        queries, vectors = await self.rewriter.rewrite(question, self.memory)

        raw_chunks = self._retrieve_and_deduplicate(queries, vectors)
        logger.info("After dedup: %d chunks", len(raw_chunks))      
        compressed = await self.compressor.compress(
            query=queries[0],
            query_vector=vectors[0],
            chunks=raw_chunks,
        )
        logger.info("After compression: %d chunks", len(compressed))  
        compressed = compressed[: self.config.top_k_final]
        logger.info("After top_k slice: %d chunks", len(compressed)) 
        return compressed, queries

    def _retrieve_and_deduplicate(
        self,
        queries: list[str],
        vectors: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k for every query variant.
        Deduplicate by (path, start_line) so the same physical chunk appears once.
        """
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for query_text, vec in zip(queries, vectors):
            results = self.vector_store.search(
                text_query=query_text,
                query_vector=vec,
                top_k=self.config.top_k_retrieval,
            )

            for r in results:
                # metadata is a plain dict from VectorRecord.metadata
                meta: dict = r.get("metadata") or {}
                path = (
                    meta.get("symbol_path")
                    or meta.get("filepath")      # ← TextChunker uses this
                    or meta.get("path")
                    or "unknown"
                        )
                start = meta.get("start_line") or meta.get("char_start") or ""
                key = f"{path}:{start}"
                if key not in seen:
                    seen.add(key)
                    r["text"] = r.get("document", "")
                    merged.append(r)

        logger.info(
            "Retrieved %d unique chunks across %d query variants.",
            len(merged),
            len(vectors),
        )
        return merged

    async def _synthesize(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        context = self._build_context_string(chunks)
        history = self.memory.format_as_messages()
        system_prompt = self._pick_system_prompt(chunks)   # ← dynamic

        messages = [
            *history,
            {"role": "user", "content": self._user_prompt(question, context)},
        ]
        return await self.client.generate(
            system_prompt=system_prompt,
            messages=messages,
        )

    def _build_context_string(self, chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            meta: dict = chunk.get("metadata") or {}
            # Prefer compressed_text if ContextCompressor trimmed the chunk
            text = chunk.get("compressed_text") or chunk.get("text", "")
            text = text.strip()
            path = (
                meta.get("symbol_path")
                or meta.get("filepath")      # ← TextChunker stores it here
                or meta.get("path")
                or "unknown"
                    )
            start = meta.get("start_line") or meta.get("page_number") or "?"
            end   = meta.get("end_line")   or meta.get("page_number") or "?"
            lang = meta.get("language") or meta.get("modality") or ""
            fence = lang if lang not in ("text", "markdown", "") else ""
            parts.append(
                f"[{i}] {path}  (lines {start}–{end})\n"
                f"```{fence}\n{text}\n```"
            )
        return "\n\n".join(parts)
    
    def _pick_system_prompt(self, chunks: list[dict[str, Any]]) -> str:
        """Choose system prompt based on majority source type in retrieved chunks."""
        counts = {"code": 0, "pdf": 0, "web": 0}
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            source_type = meta.get("source_type", "")
            if source_type == "code":
                counts["code"] += 1
            elif source_type == "pdf":
                counts["pdf"] += 1
            elif source_type == "web":
                counts["web"] += 1

        dominant = max(counts, key=counts.get)
        if counts[dominant] == 0:
            return self.config.system_prompt_default

        return {
            "code": self.config.system_prompt_code,
            "pdf":  self.config.system_prompt_pdf,
            "web":  self.config.system_prompt_web,
        }[dominant]

    def _user_prompt(self, question: str, context: str) -> str:
        return f"Context from the codebase:\n\n{context}\n\nQuestion: {question}"

    def _extract_sources(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata")
            if meta is None:
                continue
            sources.append(dict(meta) if isinstance(meta, dict) else meta.model_dump())
        return sources