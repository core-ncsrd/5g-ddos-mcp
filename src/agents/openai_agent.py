"""OpenAI function-calling agent.

Works with any OpenAI-compatible endpoint:
  • OpenAI API  (LLM_BACKEND=openai)
  • Ollama      (LLM_BACKEND=ollama  — uses its /v1/chat/completions endpoint)
  • vLLM / LM Studio / Groq / Together.ai / Mistral  (LLM_BACKEND=openai_compatible)

The agent uses the ``tools`` / ``tool_calls`` schema introduced in
OpenAI's chat completion API and supported by all of the above.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import AgentTurn, BaseAgent, ToolCall
from .registry import TOOL_SCHEMAS_OPENAI, execute_tool

logger = logging.getLogger(__name__)

# Default models per logical backend
_DEFAULT_MODELS: Dict[str, str] = {
    "openai":            "gpt-4o",
    "ollama":            "llama3.1",
    "openai_compatible": "gpt-4o",
}

# Default base_url overrides
_DEFAULT_BASE_URLS: Dict[str, Optional[str]] = {
    "openai":            None,                        # official API
    "ollama":            "http://localhost:11434/v1",
    "openai_compatible": None,                        # read from OPENAI_BASE_URL env
}


class OpenAIAgent(BaseAgent):
    """Agent backed by any OpenAI-compatible chat completion API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key:  Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc

        self._base_url = base_url
        self._api_key  = api_key

    @classmethod
    def for_backend(cls, backend: str, model: Optional[str] = None, **kwargs: Any) -> "OpenAIAgent":
        """Construct an OpenAIAgent pre-configured for a named backend."""
        resolved_model    = model or _DEFAULT_MODELS.get(backend, "gpt-4o")
        resolved_base_url = _DEFAULT_BASE_URLS.get(backend)
        if backend == "openai_compatible":
            resolved_base_url = os.getenv("OPENAI_BASE_URL", resolved_base_url)
        return cls(model=resolved_model, base_url=resolved_base_url, **kwargs)

    @property
    def framework_label(self) -> str:
        label = "OpenAI function-calling"
        if self._base_url:
            label += f"  ({self._base_url})"
        return label

    async def run_turn(self, user_message: str) -> AgentTurn:
        import openai

        client = openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key or os.getenv("OPENAI_API_KEY", "ollama"),
        )

        # Prepend system prompt as first message if history is empty
        messages: List[Dict[str, Any]] = []
        if not self.history:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        self.history.append({"role": "user", "content": user_message})

        all_tool_calls: List[ToolCall] = []
        final_text = ""

        for iteration in range(self.MAX_ITERATIONS):
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS_OPENAI,
                tool_choice="auto",
                max_tokens=4096,
            )

            choice  = response.choices[0]
            message = choice.message

            final_text = message.content or ""

            if not message.tool_calls or choice.finish_reason == "stop":
                messages.append({"role": "assistant", "content": final_text})
                self.history.append({"role": "assistant", "content": final_text})
                break

            # Append the assistant message with tool_calls to the running context
            messages.append(message.model_dump())
            self.history.append(message.model_dump())

            # Execute tool calls
            for raw_tc in message.tool_calls:
                try:
                    args = json.loads(raw_tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                tc = ToolCall(id=raw_tc.id, name=raw_tc.function.name, args=args)
                try:
                    tc.result = await execute_tool(raw_tc.function.name, args)
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", raw_tc.function.name, exc)
                    tc.result = {"error": str(exc)}

                all_tool_calls.append(tc)

                tool_msg = {
                    "role":         "tool",
                    "tool_call_id": raw_tc.id,
                    "content":      json.dumps(tc.result, default=str),
                }
                messages.append(tool_msg)
                self.history.append(tool_msg)

        else:
            logger.warning("OpenAIAgent: reached max iterations (%d)", self.MAX_ITERATIONS)

        return AgentTurn(response_text=final_text, tool_calls=all_tool_calls)
