from __future__ import annotations

import re
from typing import List, Dict, Any

import numpy as np
from pydantic import BaseModel, Field


class CompressionConfig(BaseModel):
    token_budget: int = 1200
    min_sentence_score: float =  -5.0   
    chunk_keep_ratio: float = 0.5
    max_sentences_per_chunk: int = 8
    redundancy_threshold: float = 0.92


class ContextCompressor:

    def __init__(self,reranker,config: CompressionConfig | None = None,):
        self.reranker = reranker
        self.config = config or CompressionConfig()


    async def compress(self,query: str,query_vector: np.ndarray,chunks: List[Dict[str, Any]],) -> List[Dict[str, Any]]:

            if not chunks:
                return []


            reranked = self._rerank_chunks(query, chunks)

            keep_n = max(
                1,
                int(len(reranked) * self.config.chunk_keep_ratio),
            )

            reranked = reranked[:keep_n]


            compressed = []

            for chunk in reranked:
                compressed_text = self._extract_relevant_sentences(
                    query,
                    chunk["text"],
                )

                if compressed_text.strip():
                    chunk["compressed_text"] = compressed_text
                    compressed.append(chunk)


            compressed = self._deduplicate(compressed)


            compressed = self._fit_token_budget(compressed)

            return compressed
    
    def _rerank_chunks(self,query: str,chunks: List[Dict[str, Any]],) -> List[Dict[str, Any]]:

        return self.reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=len(chunks),
        )
    
    def _extract_relevant_sentences(self, query: str, text: str) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:500]  # fallback
    
        # Drop fragments too short to be meaningful
        sentences = [s for s in sentences if len(s.split()) >= 5]
    
        if not sentences:
            return text[:500]  # fallback if all sentences are fragments
    
        scores = self.reranker.score_sentences(query=query, sentences=sentences)
        ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    
        selected = []
        for sentence, score in ranked:
            if score < self.config.min_sentence_score:
                continue
            selected.append(sentence)
            if len(selected) >= self.config.max_sentences_per_chunk:
                break
            
        if not selected:
            selected = [s for s, _ in ranked[:3]]
    
        return "\n".join(selected)
    
    def _deduplicate(self,chunks: List[Dict[str, Any]],) -> List[Dict[str, Any]]:
        unique = []
        for chunk in chunks:
            text = chunk["compressed_text"]
            is_duplicate = False
            for existing in unique:
                similarity = self._jaccard_similarity(
                    text,
                    existing["compressed_text"],
                )
                if similarity >= self.config.redundancy_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(chunk)
        return unique
    
    def _fit_token_budget(self,chunks: List[Dict[str, Any]],) -> List[Dict[str, Any]]:
        total_tokens = 0
        final_chunks = []

        for chunk in chunks:

            text = chunk["compressed_text"]

            approx_tokens = self._estimate_tokens(text)

            if total_tokens + approx_tokens > self.config.token_budget:
                break

            total_tokens += approx_tokens
            final_chunks.append(chunk)

        return final_chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        lines = [
            x.strip()
            for x in re.split(r"(?<=[.!?])\s+|\n+", text)
            if x.strip()
        ]

        return lines

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def _jaccard_similarity(self,a: str,b: str,) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)