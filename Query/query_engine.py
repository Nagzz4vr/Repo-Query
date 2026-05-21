from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

import anthropic
import numpy as np
from pydantic import BaseModel, Field

from Embedder.embedder import Embedder
from Query_Engine.context_compressor import CompressionConfig, ContextCompressor
from Query_Engine.memory import ConversationMemory, MemoryConfig
from Query_Engine.query_rewriter import QueryRewriter, RewriteConfig
from Vector_Store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryConfig(BaseModel):
    top_k_retrieval: int = Field(
        default=10,
        description="Chunks fetched per query variant before compression",
    )
    top_k_final: int = Field(
        default=5,
        description="Chunks passed to synthesis after compression",
    )
    model: str = Field(default="claude-sonnet-4-20250514")
    system_prompt: str = Field(
        default=(
            "You are an expert code assistant. Answer questions about the codebase "
            "using ONLY the provided context snippets. "
            "Always cite the source file path and line numbers when referencing code. "
            "If the answer cannot be determined from the context, say so clearly."
        )
    )


class QueryResult(BaseModel):
    question: str
    rewritten_queries: list[str]
    answer: str
    sources: list[dict[str, Any]]   # ChunkMetadata dicts for cited chunks


class QueryEngine:
    """
    Full RAG query pipeline with query rewriting, multi-vector retrieval,
    context compression, LLM synthesis, and persistent conversation memory.

    Pipeline
    --------
    user query
        → evict old memory turns (summarise if needed)
        → contextualise + expand into N query variants
        → embed all variants
        → retrieve top-k chunks per variant from vector store
        → deduplicate across variants
        → compress (similarity filter + LLM extraction)
        → synthesize answer (with memory injected as chat history)
        → update memory
        → return QueryResult
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        anthropic_api_key: Optional[str] = None,
        query_config: Optional[QueryConfig] = None,
        rewrite_config: Optional[RewriteConfig] = None,
        compression_config: Optional[CompressionConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = query_config or QueryConfig()

        self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

        self.memory = ConversationMemory(memory_config)
        self.rewriter = QueryRewriter(self.client, embedder, rewrite_config)
        self.compressor = ContextCompressor(self.client, compression_config)

    # ── public API ────────────────────────────────────────────────────────

    async def query(self, question: str) -> QueryResult:
        """Blocking pipeline — returns full QueryResult."""
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
        Memory is updated after the stream finishes.
        """
        chunks, queries = await self._run_pipeline(question)
        context = self._build_context_string(chunks)
        history = self.memory.format_as_messages()

        collected: list[str] = []

        async with self.client.messages.stream(
            model=self.config.model,
            max_tokens=2048,
            system=self.config.system_prompt,
            messages=[
                *history,
                {"role": "user", "content": self._user_prompt(queries[0], context)},
            ],
        ) as stream:
            async for token in stream.text_stream:
                collected.append(token)
                yield token

        full_answer = "".join(collected)
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", full_answer)

    def reset_memory(self) -> None:
        """Clear conversation history (e.g. start a new session)."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    # ── pipeline ──────────────────────────────────────────────────────────

    async def _run_pipeline(
        self, question: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Shared pre-synthesis steps used by both query() and stream_query()."""

        # 1. Evict stale memory if window is full
        await self.memory.evict_if_needed(self.client)

        # 2. Rewrite → (queries, vectors)
        queries, vectors = await self.rewriter.rewrite(question, self.memory)

        # 3. Multi-vector retrieval + deduplication
        raw_chunks = self._retrieve_and_deduplicate(queries, vectors)

        # 4. Compress
        compressed = await self.compressor.compress(
            query=queries[0],
            query_vector=vectors[0],
            chunks=raw_chunks,
        )
        compressed = compressed[: self.config.top_k_final]

        return compressed, queries

    # ── retrieval ─────────────────────────────────────────────────────────

    def _retrieve_and_deduplicate(
        self,
        queries: list[str],
        vectors: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k for every query variant and deduplicate by
        (path, start_line) — the same physical chunk should appear once.
        """
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []

        for query_text, vec in zip(queries, vectors):
            results = self.vector_store.search(vec, top_k=self.config.top_k_retrieval)

            for r in results:
                meta = r.get("metadata")
                path = getattr(meta, "path", None) or "unknown"
                start = getattr(meta, "start_line", None) or ""
                key = f"{path}:{start}"

                if key not in seen:
                    seen.add(key)
                    # Attach plain text so the compressor can read it
                    r["text"] = getattr(meta, "chunk_name", "") or ""
                    merged.append(r)

        logger.info(
            "Retrieved %d unique chunks across %d query variants.",
            len(merged), len(vectors),
        )
        return merged

    # ── synthesis ─────────────────────────────────────────────────────────

    async def _synthesize(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        context = self._build_context_string(chunks)
        history = self.memory.format_as_messages()

        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=2048,
            system=self.config.system_prompt,
            messages=[
                *history,
                {"role": "user", "content": self._user_prompt(question, context)},
            ],
        )
        return response.content[0].text

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_context_string(self, chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata")
            text = chunk.get("text", "").strip()
            path = getattr(meta, "path", "unknown") if meta else "unknown"
            start = getattr(meta, "start_line", "?") if meta else "?"
            end = getattr(meta, "end_line", "?") if meta else "?"
            lang = getattr(meta, "language", "") if meta else ""
            fence = lang or ""
            parts.append(
                f"[{i}] {path}  (lines {start}–{end})\n"
                f"```{fence}\n{text}\n```"
            )
        return "\n\n".join(parts)

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
            sources.append(
                meta.model_dump() if hasattr(meta, "model_dump") else dict(meta)
            )
        return sources