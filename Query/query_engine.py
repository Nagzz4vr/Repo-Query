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

class QueryConfig(BaseModel):
    top_k_retrieval: int = Field(
        default=10,
        description="Chunks fetched per query variant before reranking/compression.",
    )
    top_k_final: int = Field(
        default=5,
        description="Chunks passed to LLM synthesis after compression.",
    )
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

        messages = [
            *history,
            {"role": "user", "content": self._user_prompt(queries[0], context)},
        ]

        collected: list[str] = []
        async for token in self.client.stream(
            system_prompt=self.config.system_prompt,
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

        compressed = await self.compressor.compress(
            query=queries[0],
            query_vector=vectors[0],
            chunks=raw_chunks,
        )
        compressed = compressed[: self.config.top_k_final]
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
                path = meta.get("symbol_path") or meta.get("path") or "unknown"
                start = meta.get("start_line") or ""
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

        messages = [
            *history,
            {"role": "user", "content": self._user_prompt(question, context)},
        ]

        return await self.client.generate(
            system_prompt=self.config.system_prompt,
            messages=messages,
        )

    def _build_context_string(self, chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            meta: dict = chunk.get("metadata") or {}
            # Prefer compressed_text if ContextCompressor trimmed the chunk
            text = chunk.get("compressed_text") or chunk.get("text", "")
            text = text.strip()
            path = meta.get("symbol_path") or meta.get("path") or "unknown"
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            lang = meta.get("language") or meta.get("modality") or ""
            fence = lang if lang not in ("text", "markdown", "") else ""
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
            sources.append(dict(meta) if isinstance(meta, dict) else meta.model_dump())
        return sources