from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from Embedder.embedder import (
    Embedder,
    EmbeddingPipeline,
    RetrievalChunk,
    RetrievalModality,
)
from LLM.llm_client import LiteLLMClient
from Memory.memory import ConversationMemory

logger = logging.getLogger(__name__)

class RewriteConfig(BaseModel):
    strategy: Literal["contextual", "multi_query", "hyde", "step_back"] = Field(
        default="multi_query",
        description=(
            "contextual  — standalone rewrite using chat history only\n"
            "multi_query — N diverse variants for higher recall\n"
            "hyde        — embed a hypothetical answer doc alongside the query\n"
            "step_back   — original + a generalised fallback query"
        ),
    )
    n_variants: int = Field(default=3, ge=1, le=8)


class QueryRewriter:

    def __init__(self,client: LiteLLMClient,embedder: Embedder,config: Optional[RewriteConfig] = None) -> None:
        self.client = client
        self.pipeline = EmbeddingPipeline(embedder)
        self.config = config or RewriteConfig()

    async def rewrite(self,query: str,memory: ConversationMemory) -> tuple[list[str], list[np.ndarray]]:
        if not memory.is_empty():
            query = await self._contextualize(query, memory)
            logger.debug("Contextualised query: %s", query)

        match self.config.strategy:
            case "multi_query":
                queries = await self._multi_query(query)
            case "hyde":
                queries = await self._hyde(query)
            case "step_back":
                queries = [query] + await self._step_back(query)
            case _:  # "contextual" — single contextualised query
                queries = [query]

        vectors = self._embed_queries(queries)

        logger.info(
            "Rewrite: strategy=%s  total_queries=%d",
            self.config.strategy,
            len(queries),
        )
        return queries, vectors
    
    async def _contextualize(self, query: str, memory: ConversationMemory) -> str:
        history = memory.format_as_text()
        prompt = (
            "Given the conversation history below, rewrite the latest question "
            "as a fully self-contained question that can be understood without "
            "any prior context. Return ONLY the rewritten question.\n\n"
            f"History:\n{history}\n\n"
            f"Latest question: {query}\n\nStandalone question:"
        )
        return await self._complete(prompt)

    async def _multi_query(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self.config.n_variants} different search queries to help "
            "retrieve relevant code or documentation for the question below. "
            "Use different phrasings, abstraction levels, and perspectives. "
            "Return one query per line, no numbering, no extra text.\n\n"
            f"Question: {query}"
        )
        raw = await self._complete(prompt)
        variants = [q.strip() for q in raw.splitlines() if q.strip()]
        # original query is always first; append LLM variants up to n_variants
        return [query] + variants[: self.config.n_variants]

    async def _hyde(self, query: str) -> list[str]:
        """
        Hypothetical Document Embedding: generate a plausible answer snippet
        and embed it alongside the original query.
        """
        prompt = (
            "Write a short, plausible Python code snippet or documentation "
            "passage that would directly answer the following question. "
            "Return ONLY the snippet or passage, nothing else.\n\n"
            f"Question: {query}"
        )
        hypothetical = await self._complete(prompt)
        return [query, hypothetical]

    async def _step_back(self, query: str) -> list[str]:
        prompt = (
            "Rewrite the following question as a more general, higher-level "
            "question that captures the underlying concept or task. "
            "Return ONLY the rewritten question.\n\n"
            f"Question: {query}"
        )
        general = await self._complete(prompt)
        return [general]

    def _embed_queries(self, queries: list[str]) -> list[np.ndarray]:
        """
        Embed all query variants in one batched pass via EmbeddingPipeline.
        Each query string becomes a TEXT-modality RetrievalChunk.
        """
        chunks = [
            RetrievalChunk(
                chunk_id=str(i),
                modality=RetrievalModality.TEXT,
                text=q,
            )
            for i, q in enumerate(queries)
        ]
        records = self.pipeline.run(chunks)
        return [np.asarray(r.vector) for r in records]

    async def _complete(self, prompt: str) -> str:
        return await self.client.generate(
            system_prompt="You are a helpful assistant for code search and retrieval.",
            messages=[{"role": "user", "content": prompt}],
        )