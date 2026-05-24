from __future__ import annotations

import os
import asyncio
import logging
from typing import AsyncIterator, Optional
import time
from litellm import completion

from litellm.exceptions import (
    RateLimitError,
    NotFoundError,
    ServiceUnavailableError,
)
from Tracker.token_ledger import TokenLedger
logger = logging.getLogger(__name__)


class LiteLLMClient:

    def __init__(self,model_pool: list[str],request_id: str,
        session_id: str,log_dir: str = "token_ledger/",temperature: float = 0.1,max_tokens: int = 2048,retry_after: int = 10,):
        self.model_pool = model_pool
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_after = retry_after
        self._model_index = 0
        self.ledger = TokenLedger(
            request_id=request_id,
            session_id=session_id,
            log_dir=log_dir,
        )
        self.user_id     = None

        self.api_key    = os.environ["GROQ_API_KEY"]

    @property
    def current_model(self) -> str:
        return self.model_pool[self._model_index]

    def rotate_model(self):

        self._model_index = (
            self._model_index + 1
        ) % len(self.model_pool)

        logger.warning(
            "Rotated model → %s",
            self.current_model,
        )


    async def generate(self,system_prompt: str,messages: list[dict],) -> str:

        last_error = None
        retry_count = 0

        for _ in range(len(self.model_pool)):

            start = time.perf_counter()

            try:
                self.agent_id=self.current_model
                response = await asyncio.to_thread(
                    completion,
                    model=self.current_model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        *messages,
                    ],
                )

                latency_ms = int(
                    (time.perf_counter() - start) * 1000
                )

                usage = response.usage

                prompt_tokens = getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )

                completion_tokens = getattr(
                    usage,
                    "completion_tokens",
                    0,
                )

                self.ledger.record(
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    model=self.current_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    success=True,
                )

                return response.choices[0].message.content

            except (
                RateLimitError,
                ServiceUnavailableError,
            ) as e:

                latency_ms = int(
                    (time.perf_counter() - start) * 1000
                )

                logger.warning(
                    "Model failed: %s",
                    self.current_model,
                )

                self.ledger.record(
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    model=self.current_model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    success=False,
                    error=str(e),
                )

                last_error = e
                retry_count += 1

                self.rotate_model()

                await asyncio.sleep(self.retry_after)

            except NotFoundError as e:

                self.ledger.record(
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    model=self.current_model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    retry_count=retry_count,
                    success=False,
                    error=str(e),
                )

                retry_count += 1

                self.rotate_model()

        raise RuntimeError(
            f"All models failed: {last_error}"
        )
    
    
    async def stream(self,system_prompt: str,messages: list[dict],) -> AsyncIterator[str]:
            start = time.perf_counter()
            response = await asyncio.to_thread(
                completion,
                model=self.current_model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    *messages,
                ],
            )
            full_response = []

            prompt_tokens = 0
            completion_tokens = 0
            for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = getattr(
                        chunk.usage,
                        "prompt_tokens",
                        0,
                    )
                    completion_tokens = getattr(
                        chunk.usage,
                        "completion_tokens",
                        0,
                    )
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_response.append(delta.content)
                    yield delta.content
            latency_ms = int(
                (time.perf_counter() - start) * 1000
            )
            self.ledger.record(
                user_id=self.user_id,
                agent_id=self.agent_id,
                model=self.current_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=True,)

