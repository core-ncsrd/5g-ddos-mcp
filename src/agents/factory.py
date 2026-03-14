"""Agent factory.

Selects the right agent backend based on the ``framework`` parameter
and/or the active ``LLM_BACKEND`` environment variable.

Auto-selection order when ``framework="auto"``:
  1. LLM_BACKEND=anthropic  → AnthropicAgent
  2. LLM_BACKEND=ollama     → OpenAIAgent (Ollama's /v1 endpoint)
  3. LLM_BACKEND=openai or openai_compatible → OpenAIAgent
  4. Fallback               → AnthropicAgent (requires ANTHROPIC_API_KEY)

Explicit ``framework`` values: ``anthropic`` | ``openai`` | ``ollama``
                               | ``langchain`` | ``langgraph``
(``langchain`` and ``langgraph`` are aliases for the LangGraph ReAct agent.)
"""
from __future__ import annotations

import os
from typing import Optional

from .base import BaseAgent


def get_agent(
    framework: str = "auto",
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
) -> BaseAgent:
    """Return a fully configured agent for the requested framework.

    Parameters
    ----------
    framework:
        One of ``auto`` | ``anthropic`` | ``openai`` | ``ollama``
        | ``openai_compatible`` | ``langchain`` | ``langgraph``.
        ``"auto"`` picks the best available agent based on LLM_BACKEND.
    model:
        Override the default model name for the selected backend.
    system_prompt:
        Replace the default system prompt.
    verbose:
        If True, the REPL will show extra debug output.
    """
    backend = os.getenv("LLM_BACKEND", "anthropic").lower()

    # Resolve "auto" using the environment
    if framework == "auto":
        framework = backend

    kwargs = dict(verbose=verbose)
    if system_prompt:
        kwargs["system_prompt"] = system_prompt

    # ── Anthropic ─────────────────────────────────────────────────────────────
    if framework == "anthropic":
        from .anthropic_agent import AnthropicAgent
        return AnthropicAgent(
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            **kwargs,
        )

    # ── OpenAI (official API) ─────────────────────────────────────────────────
    if framework == "openai":
        from .openai_agent import OpenAIAgent
        return OpenAIAgent.for_backend(
            "openai",
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
            **kwargs,
        )

    # ── Ollama (local, OpenAI-compat /v1) ─────────────────────────────────────
    if framework == "ollama":
        from .openai_agent import OpenAIAgent
        return OpenAIAgent.for_backend(
            "ollama",
            model=model or os.getenv("OLLAMA_MODEL", "llama3.1"),
            **kwargs,
        )

    # ── Generic OpenAI-compatible endpoint ────────────────────────────────────
    if framework == "openai_compatible":
        from .openai_agent import OpenAIAgent
        return OpenAIAgent.for_backend(
            "openai_compatible",
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
            **kwargs,
        )

    # ── LangGraph ReAct (langchain / langgraph aliases) ───────────────────────
    if framework in ("langchain", "langgraph"):
        from .langchain_agent import LangGraphAgent
        # Use the underlying LLM_BACKEND as the model provider inside LangGraph
        lg_backend = backend if backend in ("anthropic", "openai", "openai_compatible", "ollama") else "anthropic"
        return LangGraphAgent(backend=lg_backend, model=model, **kwargs)

    raise ValueError(
        f"Unknown agent framework: {framework!r}. "
        "Choose: auto | anthropic | openai | ollama | openai_compatible | langchain | langgraph"
    )
