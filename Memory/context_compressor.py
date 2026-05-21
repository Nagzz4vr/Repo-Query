from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import anthropic
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CompressionConfig(BaseModel):
    strategy: Literal["similarity_filter", "llm_extract", "both"] = Field(
        default="both",
        description=(
            "similarity_filter — drop chunks below cosine-similarity threshold\n"
            "llm_extract       — LLM keeps only sentences relevant to the query\n"
            "both              — filter first (cheap), then extract (thorough)"
        ),
    )
    similarity_threshold: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Min cosine similarity to survive the filter step",
    )
    max_chunks: int = Field(default=6, ge=1, description="Hard cap after compression")
    max_extracted_chars: int = Field(
        default=800, description="Char limit per chunk after LLM extraction"
    )
    model: str = Field(default="claude-sonnet-4-20250514")


class ContextCompressor:
    """
    Two-stage context compression pipeline.

    Stage 1 — similarity_filter:
        Re-score retrieved chunks against the query vector and drop any whose
        cosine similarity falls below the threshold. Fast, no LLM calls.

    Stage 2 — llm_extract:
        For each surviving chunk, ask the LLM to return only the sentences
        that are directly relevant to the query. Chunks with nothing relevant
        are dropped entirely.

    Running both stages (strategy="both") is the recommended default:
        - Filter removes obviously off-topic chunks cheaply.
        - Extract trims the survivors, reducing synthesis prompt size.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        config: Optional[CompressionConfig] = None,
    ) -> None:
        self.client = client
        self.config = config or CompressionConfig()

    async def compress(
        self,
        query: str,
        query_vector: np.ndarray,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        query        : the (contextualised) user question
        query_vector : unit-normalised embedding of query
        chunks       : output of _retrieve_and_deduplicate()
                       each dict must contain "text", "metadata", "score"
                       and optionally "vector" (for similarity re-scoring)

        Returns
        -------
        Compressed, ordered list of chunk dicts capped at max_chunks.
        """
        if not chunks:
            return []

        result = list(chunks)

        if self.config.strategy in ("similarity_filter", "both"):
            result = self._similarity_filter(query_vector, result)

        if self.config.strategy in ("llm_extract", "both"):
            result = await self._llm_extract(query, result)

        return result[: self.config.max_chunks]

    # ── Stage 1 ───────────────────────────────────────────────────────────

    def _similarity_filter(
        self,
        query_vector: np.ndarray,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Re-rank by cosine similarity (vectors are unit-normalised so dot = cosine).
        Fall back to FAISS score when the raw vector is absent.
        """
        scored: list[tuple[float, dict]] = []

        for chunk in chunks:
            vec = chunk.get("vector")
            if vec is not None:
                sim = float(np.dot(query_vector, np.asarray(vec)))
            else:
                # FAISS flat_l2 score is squared L2 distance; convert roughly
                # This is an approximation — prefer storing vectors when possible
                faiss_score = chunk.get("score", 0.0)
                sim = max(0.0, 1.0 - faiss_score / 2.0)

            if sim >= self.config.similarity_threshold:
                chunk = dict(chunk)
                chunk["similarity"] = sim
                scored.append((sim, chunk))

        scored.sort(key=lambda t: t[0], reverse=True)
        kept = [c for _, c in scored]

        logger.debug(
            "Similarity filter: %d → %d chunks (threshold=%.2f)",
            len(chunks), len(kept), self.config.similarity_threshold,
        )
        return kept

    # ── Stage 2 ───────────────────────────────────────────────────────────

    async def _llm_extract(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Ask the LLM to extract only relevant sentences from each chunk.
        Chunks whose entire content is irrelevant are dropped.
        """
        compressed: list[dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_extracted_chars // 3,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract only the lines or sentences from the passage "
                            "below that are directly relevant to answering the query. "
                            "Preserve code exactly as written. "
                            "If nothing is relevant respond with exactly: IRRELEVANT\n\n"
                            f"Query: {query}\n\n"
                            f"Passage:\n{text}\n\n"
                            "Relevant content:"
                        ),
                    }
                ],
            )
            extracted = response.content[0].text.strip()

            if not extracted or extracted.upper() == "IRRELEVANT":
                logger.debug("Chunk dropped as irrelevant by LLM extractor.")
                continue

            new_chunk = dict(chunk)
            new_chunk["text"] = extracted[: self.config.max_extracted_chars]
            new_chunk["compressed"] = True
            compressed.append(new_chunk)

        logger.debug(
            "LLM extraction: %d → %d chunks", len(chunks), len(compressed)
        )
        return compressed