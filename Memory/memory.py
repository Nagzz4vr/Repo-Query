from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class LLMClientProtocol(Protocol):
    """Protocol to abstract away specific model providers (Anthropic, OpenAI, etc.)"""
    async def generate(self, messages: List[ChatMessage], max_tokens: int) -> str: ...


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0

    def serialize(self) -> Dict[str, Any]:
        """Canonical dictionary representation matching standard API shapes."""
        return {"role": self.role, "content": f"[{self.timestamp.isoformat()}] {self.content}"}

class SummaryProvenance(BaseModel):
    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_turn_ids: List[str] = Field(default_factory=list)
    lineage_depth: int = 0
    raw_hash: str

class MemorySnapshot(BaseModel):
    """Deterministic, immutable snapshot for durability, auditing, or replays."""
    summary: Optional[str]
    provenance: Optional[SummaryProvenance]
    turns: List[Turn]

class MemoryConfig(BaseModel):
    # Budgeting based on actual data weight instead of loose cardinality count
    max_verbatim_tokens: int = 4000 
    target_eviction_tokens: int = 2000
    approx_chars_per_token: float = 4.0  # Safe heuristic fallback if no tokenizer callback is provided

class ConversationMemory:
    """
    Concurrency-safe, provider-agnostic sliding-window memory engine.
    Maintains semantic parity, structural lineage tracking, and token budget awareness.
    """

    def __init__(
        self, 
        config: Optional[MemoryConfig] = None,
        token_counter: Optional[Callable[[str], int]] = None
    ) -> None:
        self.config = config or MemoryConfig()
        self._token_counter = token_counter or self._default_token_heuristic
        
        self._turns: List[Turn] = []
        self._summary: Optional[str] = None
        self._provenance: Optional[SummaryProvenance] = None
        
        # Concurrency safety boundary for long-running async LLM summaries
        self._lock = asyncio.Lock()

    def _default_token_heuristic(self, text: str) -> int:
        return int(len(text) / self.config.approx_chars_per_token)


    def add_turn(self, role: Literal["user", "assistant"], content: str) -> None:
        """Appends a conversation turn with eager token accounting evaluation."""
        tokens = self._token_counter(content)
        self._turns.append(Turn(role=role, content=content, token_count=tokens))

    def is_empty(self) -> bool:
        return not self._turns and not self._summary

    def clear(self) -> None:
        self._turns.clear()
        self._summary = None
        self._provenance = None


    def format_as_messages(self) -> List[ChatMessage]:
        """
        Renders the entire timeline context natively.
        Uses system/metadata framing to pass context without fabricating conversations.
        """
        messages: List[ChatMessage] = []
        
        if self._summary:
            # Clear demarcations separate background context metadata from interactive logs
            meta_content = (
                f"BACKGROUND SYSTEM CONTEXT (Lineage Depth: {self._provenance.lineage_depth if self._provenance else 1})\n"
                f"Compressed Context Summary: {self._summary}"
            )
            messages.append(ChatMessage(role="system", content=meta_content))
            
        for turn in self._turns:
            messages.append(ChatMessage(role=turn.role, content=f"[{turn.timestamp.isoformat()}] {turn.content}"))
            
        return messages

    def format_as_text(self) -> str:
        """Canonical plain text rendering."""
        parts: List[str] = []
        if self._summary:
            parts.append(f"--- BEGIN ARCHIVED CONTEXT SUMMARY ---\n{self._summary}\n--- END ARCHIVED CONTEXT SUMMARY ---")
        for turn in self._turns:
            parts.append(f"[{turn.timestamp.isoformat()}] {turn.role.upper()}: {turn.content}")
        return "\n".join(parts)


    def get_snapshot(self) -> MemorySnapshot:
        """Produces an isolated, immutable snapshot copy of current internal states."""
        return MemorySnapshot(
            summary=self._summary,
            provenance=self._provenance.model_copy(deep=True) if self._provenance else None,
            turns=[t.model_copy(deep=True) for t in self._turns]
        )

    def restore_from_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Restores memory state deterministically from an existing snapshot."""
        self._summary = snapshot.summary
        self._provenance = snapshot.provenance
        self._turns = [t.model_copy(deep=True) for t in snapshot.turns]



    async def evict_if_needed(self, llm_client: LLMClientProtocol) -> None:
        """
        Token-budget aware eviction routine with lock-guarded concurrency protection.
        Ensures perfect user/assistant exchange boundary alignment.
        """
        current_token_total = sum(t.token_count for t in self._turns)
        if current_token_total <= self.config.max_verbatim_tokens:
            return

        # Acquire lock to wrap async state mutations securely across task switches
        async with self._lock:
            # Re-verify condition post-lock acquisition to eliminate race conditions
            if sum(t.token_count for t in self._turns) <= self.config.max_verbatim_tokens:
                return

            # Target collection slice while keeping structural exchange pairs safe
            eviction_slice_index = 0
            accumulated_tokens = 0
            
            # Group checking by pairs (User + Assistant) to avoid splitting dialogue boundaries
            for i in range(0, len(self._turns), 2):
                if i + 1 < len(self._turns):
                    pair_tokens = self._turns[i].token_count + self._turns[i+1].token_count
                    if accumulated_tokens + pair_tokens > self.config.target_eviction_tokens:
                        break
                    accumulated_tokens += pair_tokens
                    eviction_slice_index = i + 2

            if eviction_slice_index == 0:
                return # Base window bounds are too narrow to yield safe slices

            old_turns = self._turns[:eviction_slice_index]
            self._turns = self._turns[eviction_slice_index:]

            old_turns_text = "\n".join(
                f"[{t.timestamp.isoformat()}] {t.role}: {t.content}" for t in old_turns
            )
            
            # Pack historical data as explicit chronological inputs to block context drift
            compounding_prompt = "You are an analytical state tracker updating a continuous memory digest.\n"
            if self._summary:
                compounding_prompt += (
                    f"CRITICAL PRIOR STATE SUMMARY:\n{self._summary}\n\n"
                    f"Add the insights from the following recent chronological timeline turns to the existing context. "
                    f"Do not lose core technical insights, decisions made, file coordinates, or error stacks.\n"
                )
            else:
                compounding_prompt += "Synthesize a concise context tracking log from the following chronological timeline turns.\n"

            compounding_prompt += f"NEW TIMELINE TURNS TO MERGE:\n{old_turns_text}\n\nREVISED COMPREHENSIVE CONTEXT DIGEST:"

            logger.info("Executing isolated context compilation for %d turns.", len(old_turns))
            
            # Execute compilation using the provider-agnostic interface
            compiled_summary = await llm_client.generate(
                messages=[ChatMessage(role="user", content=compounding_prompt)],
                max_tokens=1024
            )
            
            # Update explicit lineage tracking state structures
            raw_content_bytes = compiled_summary.encode("utf-8")
            content_hash = hashlib.sha256(raw_content_bytes).hexdigest()
            
            prior_depth = self._provenance.lineage_depth if self._provenance else 0
            
            self._summary = compiled_summary.strip()
            self._provenance = SummaryProvenance(
                version=1,
                updated_at=datetime.now(timezone.utc),
                lineage_depth=prior_depth + 1,
                raw_hash=content_hash
            )
            
            logger.info("Context compilation successfully written. Hash: %s", content_hash[:8])