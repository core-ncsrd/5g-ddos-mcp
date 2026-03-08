"""
Unified LLM client supporting:
  - Claude (Anthropic API)
  - Ollama  (local open-source models: llama3, mistral, phi4, gemma3, …)
  - OpenAI-compatible endpoints (vLLM, LM Studio, Groq, Together.ai, …)

Usage:
    from src.models.llm_client import get_llm_client
    client = get_llm_client()
    response = await client.complete("Explain this 5G DDoS attack…")
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Return the model's text response to `prompt`."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable / configured."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Claude (Anthropic)
# ─────────────────────────────────────────────────────────────────────────────
class ClaudeClient(BaseLLMClient):
    """Calls the Anthropic Messages API (claude-sonnet-4-6 by default)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        self._api_key = api_key
        self._model   = model

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        kwargs: dict = {
            "model":      self._model,
            "max_tokens": max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        message = await client.messages.create(**kwargs)
        return message.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# Ollama  (local open-source LLMs)
# ─────────────────────────────────────────────────────────────────────────────
class OllamaClient(BaseLLMClient):
    """
    Calls a locally running Ollama server (https://ollama.com).

    Recommended models (pull with `ollama pull <name>`):
        llama3.2      – Meta Llama 3.2 (3B / 8B)
        mistral       – Mistral 7B
        phi4          – Microsoft Phi-4 (14B)
        gemma3        – Google Gemma 3 (4B / 12B / 27B)
        deepseek-r1   – DeepSeek-R1 (8B distill)
        qwen2.5       – Alibaba Qwen 2.5 (7B / 14B / 72B)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model:    str = "llama3.2",
    ):
        self._base_url = base_url.rstrip("/")
        self._model    = model

    def is_available(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        try:
            import ollama as _ollama
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            loop    = asyncio.get_event_loop()
            client  = _ollama.AsyncClient(host=self._base_url)
            resp    = await client.chat(
                model    = self._model,
                messages = messages,
                options  = {"num_predict": max_tokens, "temperature": temperature},
            )
            return resp["message"]["content"]

        except ImportError:
            # Fallback: raw HTTP if `ollama` Python package not installed
            return await self._http_fallback(prompt, system, max_tokens, temperature)

    async def _http_fallback(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        import aiohttp, json

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":    self._model,
            "messages": messages,
            "stream":   False,
            "options":  {"num_predict": max_tokens, "temperature": temperature},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                return data["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible endpoint  (vLLM, LM Studio, Groq, Together.ai, …)
# ─────────────────────────────────────────────────────────────────────────────
class OpenAICompatibleClient(BaseLLMClient):
    """
    Uses the openai SDK but points to any OpenAI-compatible base URL.

    Examples:
        vLLM server:    base_url="http://localhost:8080/v1", model="<your-model>"
        LM Studio:      base_url="http://localhost:1234/v1", api_key="lm-studio"
        Groq:           base_url="https://api.groq.com/openai/v1"
        Together.ai:    base_url="https://api.together.xyz/v1"
        Mistral API:    base_url="https://api.mistral.ai/v1"
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: str           = "https://api.openai.com/v1",
        model:    str           = "gpt-4o-mini",
    ):
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        self._api_key  = api_key or "none"
        self._base_url = base_url
        self._model    = model

    def is_available(self) -> bool:
        return True  # best-effort; actual check happens on first call

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await client.chat.completions.create(
            model       = self._model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_llm_client(config=None) -> BaseLLMClient:
    """
    Build the LLM client from config (defaults to src.config.llm_config).
    Backend priority if LLM_BACKEND is not set:
      1. Claude  (if ANTHROPIC_API_KEY is present)
      2. Ollama  (if Ollama server is reachable at OLLAMA_BASE_URL)
      3. OpenAI-compatible (fallback)
    """
    if config is None:
        from src.config import llm_config as config

    backend = config.backend

    if backend == "claude":
        if not config.anthropic_api_key:
            raise ValueError(
                "LLM_BACKEND=claude requires ANTHROPIC_API_KEY to be set."
            )
        logger.info("LLM backend: Claude (%s)", config.claude_model)
        return ClaudeClient(config.anthropic_api_key, config.claude_model)

    elif backend == "ollama":
        logger.info(
            "LLM backend: Ollama (%s @ %s)",
            config.ollama_model,
            config.ollama_base_url,
        )
        return OllamaClient(config.ollama_base_url, config.ollama_model)

    elif backend == "openai_compatible":
        logger.info(
            "LLM backend: OpenAI-compatible (%s @ %s)",
            config.openai_model,
            config.openai_base_url,
        )
        return OpenAICompatibleClient(
            config.openai_api_key,
            config.openai_base_url,
            config.openai_model,
        )

    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r}")
