"""LLM client with streaming support for multiple backends.

Supported backends:
  - openai: OpenAI streaming API (GPT-4o-mini, etc.)
  - anthropic: Anthropic streaming API (Claude 3.5 Haiku, etc.)
  - mock: Returns a canned response for testing
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import AsyncGenerator

from src.config import LLMBackend, LLMConfig
from src.pipeline.models import LLMResponse

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 10


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config
        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self._history: deque[dict] = deque(maxlen=_MAX_HISTORY_TURNS * 2)

    def add_to_history(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        if content.strip():
            self._history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history (e.g. on session reset)."""
        self._history.clear()

    def _build_messages(self, user_text: str) -> list[dict]:
        """Build the full message list including history and the new user turn."""
        messages = list(self._history)
        messages.append({"role": "user", "content": user_text})
        return messages

    @abstractmethod
    async def generate_stream(
        self, user_text: str
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream text generation from the LLM."""
        ...

    @abstractmethod
    async def generate(self, user_text: str) -> str:
        """Complete text generation (non-streaming)."""
        ...


class OpenAILLMClient(LLMClient):
    """OpenAI streaming LLM backend."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
            except ImportError:
                raise ImportError("openai package not installed: pip install openai")
        return self._client

    async def generate_stream(
        self, user_text: str
    ) -> AsyncGenerator[LLMResponse, None]:
        start = time.perf_counter()
        first_token_yielded = False
        accumulated = ""

        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self._build_messages(user_text))

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                stream_options={"include_usage": False},
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""

                    if content:
                        now = time.perf_counter()
                        if not first_token_yielded:
                            first_token_yielded = True
                            yield LLMResponse(
                                text=content,
                                is_first_token=True,
                                is_final=False,
                                latency_ms=(now - start) * 1000,
                                model=self.config.openai_model,
                            )
                        else:
                            yield LLMResponse(
                                text=content,
                                is_first_token=False,
                                is_final=False,
                                latency_ms=(now - start) * 1000,
                                model=self.config.openai_model,
                            )
                        accumulated += content

                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason:
                        total_latency = (time.perf_counter() - start) * 1000
                        if accumulated:
                            self.add_to_history("user", user_text)
                            self.add_to_history("assistant", accumulated)
                        yield LLMResponse(
                            text="",
                            is_first_token=False,
                            is_final=True,
                            latency_ms=total_latency,
                            model=self.config.openai_model,
                            finish_reason=finish_reason,
                        )

        except Exception as e:
            logger.error(f"OpenAI LLM streaming failed: {e}")
            yield LLMResponse(
                text="",
                is_final=True,
                error=str(e),
            )

    async def generate(self, user_text: str) -> str:
        full = ""
        async for chunk in self.generate_stream(user_text):
            if chunk.is_final:
                break
            full += chunk.text
        return full


class AnthropicLLMClient(LLMClient):
    """Anthropic (Claude) streaming LLM backend."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
            except ImportError:
                raise ImportError("anthropic package not installed: pip install anthropic")
        return self._client

    async def generate_stream(
        self, user_text: str
    ) -> AsyncGenerator[LLMResponse, None]:
        start = time.perf_counter()
        first_token_yielded = False
        accumulated = ""

        messages = self._build_messages(user_text)

        try:
            async with self.client.messages.stream(
                model=self.config.anthropic_model,
                system=self.config.system_prompt,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ) as stream:
                async for text_delta in stream.text_stream:
                    now = time.perf_counter()
                    accumulated += text_delta
                    if not first_token_yielded:
                        first_token_yielded = True
                        yield LLMResponse(
                            text=text_delta,
                            is_first_token=True,
                            is_final=False,
                            latency_ms=(now - start) * 1000,
                            model=self.config.anthropic_model,
                        )
                    else:
                        yield LLMResponse(
                            text=text_delta,
                            is_first_token=False,
                            is_final=False,
                            latency_ms=(now - start) * 1000,
                            model=self.config.anthropic_model,
                        )

                if accumulated:
                    self.add_to_history("user", user_text)
                    self.add_to_history("assistant", accumulated)

                total_latency = (time.perf_counter() - start) * 1000
                yield LLMResponse(
                    text="",
                    is_first_token=False,
                    is_final=True,
                    latency_ms=total_latency,
                    model=self.config.anthropic_model,
                    finish_reason="stop",
                )

        except Exception as e:
            logger.error(f"Anthropic LLM streaming failed: {e}")
            yield LLMResponse(
                text="",
                is_final=True,
                error=str(e),
            )

    async def generate(self, user_text: str) -> str:
        full = ""
        async for chunk in self.generate_stream(user_text):
            if chunk.is_final:
                break
            full += chunk.text
        return full


class MockLLMClient(LLMClient):
    """Mock LLM backend for testing without API keys.

    Returns a canned response after a configurable delay.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._mock_delay: float = 0.5  # TTFT simulation
        self._token_delay: float = 0.02  # per-token delay

    async def generate_stream(
        self, user_text: str
    ) -> AsyncGenerator[LLMResponse, None]:
        start = time.perf_counter()

        # Simulate network latency (TTFT)
        await asyncio.sleep(self._mock_delay)

        mock_response = f"Mock response to: \"{user_text[:40]}\". This is a mock LLM response."

        # Split into word-level chunks for streaming simulation
        words = mock_response.split()
        accumulated = ""

        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(self._token_delay)
            accumulated += chunk
            now = time.perf_counter()

            if i == 0:
                yield LLMResponse(
                    text=chunk,
                    is_first_token=True,
                    is_final=False,
                    latency_ms=(now - start) * 1000,
                    model="mock",
                )
            else:
                yield LLMResponse(
                    text=chunk,
                    is_first_token=False,
                    is_final=False,
                    latency_ms=(now - start) * 1000,
                    model="mock",
                )

        total_latency = (time.perf_counter() - start) * 1000
        yield LLMResponse(
            text="",
            is_first_token=False,
            is_final=True,
            latency_ms=total_latency,
            model="mock",
            finish_reason="stop",
        )

    async def generate(self, user_text: str) -> str:
        full = ""
        async for chunk in self.generate_stream(user_text):
            if chunk.is_final:
                break
            full += chunk.text
        return full


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function to create the appropriate LLM client."""
    if config.backend == LLMBackend.OPENAI:
        return OpenAILLMClient(config)
    elif config.backend == LLMBackend.ANTHROPIC:
        return AnthropicLLMClient(config)
    elif config.backend == LLMBackend.MOCK:
        return MockLLMClient(config)
    else:
        raise ValueError(f"Unknown LLM backend: {config.backend}")
