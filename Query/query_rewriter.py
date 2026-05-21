from __future__ import annotations

import logging
from typing import Literal, Optional

import anthropic
import numpy as np
from pydantic import BaseModel, Field

from Chunker.shared_models import Chunk, ChunkMethod
from Embedder.embedder import Embedder
from Query_Engine.memory import ConversationMemory

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
    model: str = Field(default="claude-sonnet-4-20250514")


class QueryRewriter:
    """
    Transforms the raw user query into one or more search queries and
    pre-computes their embedding vectors for retrieval.

    All strategies first contextualise the query (make it standalone) when
    conversation history is present, then apply the chosen expansion.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        embedder: Embedder,
        config: Optional[RewriteConfig] = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.config = config or RewriteConfig()

    async def rewrite(
        self,
        query: str,
        memory: ConversationMemory,
    ) -> tuple[list[str], list[np.ndarray]]:
        """
        Returns (queries, vectors).
        The first element is always the (possibly contextualised) original query.
        """
        # 1. Contextualise if there is prior history
        if not memory.is_empty():
            query = await self._contextualize(query, memory)
            logger.debug("Contextualised: %s", query)

        # 2. Expand
        match self.config.strategy:
            case "multi_query":
                queries = await self._multi_query(query)
            case "hyde":
                queries = await self._hyde(query)
            case "step_back":
                queries = [query] + await self._step_back(query)
            case _:   # "contextual" — single contextualised query
                queries = [query]

        # 3. Embed all variants in one pass
        vectors = self._embed_queries(queries)

        logger.info(
            "Rewrite: strategy=%s  total_queries=%d", self.config.strategy, len(queries)
        )
        return queries, vectors

    # ── strategies ────────────────────────────────────────────────────────

    async def _contextualize(self, query: str, memory: ConversationMemory) -> str:
        """Turn a follow-up question into a fully self-contained question."""
        history = memory.format_as_text()
        prompt = (
            "Given the conversation history below, rewrite the latest question "
            "as a fully self-contained question that can be understood without "
            "any prior context. Return ONLY the rewritten question.\n\n"
            f"History:\n{history}\n\n"
            f"Latest question: {query}\n\n"
            "Standalone question:"
        )
        return await self._complete(prompt, max_tokens=200)

    async def _multi_query(self, query: str) -> list[str]:
        """
        Generate N semantically diverse variants.
        Different angles → better recall across chunked code.
        """
        prompt = (
            f"Generate {self.config.n_variants} different search queries to help "
            "retrieve relevant code or documentation for the question below. "
            "Use different phrasings, abstraction levels, and perspectives. "
            "Return one query per line, no numbering, no extra text.\n\n"
            f"Question: {query}"
        )
        raw = await self._complete(prompt, max_tokens=400)
        variants = [q.strip() for q in raw.splitlines() if q.strip()]
        return [query] + variants[: self.config.n_variants]

    async def _hyde(self, query: str) -> list[str]:
        """
        Hypothetical Document Embedding:
        Generate a plausible code snippet / doc passage that would answer the
        query, then embed it alongside the original.
        The synthetic doc lands in the same vector space as real chunks.
        """
        prompt = (
            "Write a short, plausible Python code snippet or documentation "
            "passage that would directly answer the following question. "
            "Return ONLY the snippet or passage, nothing else.\n\n"
            f"Question: {query}"
        )
        hypothetical = await self._complete(prompt, max_tokens=500)
        return [query, hypothetical]

    async def _step_back(self, query: str) -> list[str]:
        """
        Return a broader, more conceptual version of the query.
        Useful when the specific phrasing might miss indirectly relevant chunks.
        """
        prompt = (
            "Rewrite the following question as a more general, higher-level "
            "question that captures the underlying concept or task. "
            "Return ONLY the rewritten question.\n\n"
            f"Question: {query}"
        )
        general = await self._complete(prompt, max_tokens=150)
        return [general]

    # ── helpers ───────────────────────────────────────────────────────────

    def _embed_queries(self, queries: list[str]) -> list[np.ndarray]:
        """Reuse the Embedder by wrapping strings in minimal Chunk objects."""
        fake_chunks = [
            Chunk(
                chunk_index=i,
                page_number=0,
                text=q,
                method=ChunkMethod.SEMANTIC,
            )
            for i, q in enumerate(queries)
        ]
        return [self.embedder.embed_chunk(c) for c in fake_chunks]

    async def _complete(self, prompt: str, max_tokens: int) -> str:
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()