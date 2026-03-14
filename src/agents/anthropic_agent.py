"""Anthropic tool_use agent.

Uses the Anthropic Messages API with native ``tool_use`` / ``tool_result``
content blocks — no extra framework required.

Supports all Claude models that have tool use (claude-3-*, claude-sonnet-4-6, etc.).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from .base import AgentTurn, BaseAgent, ToolCall
from .registry import execute_tool, get_anthropic_schemas

logger = logging.getLogger(__name__)


class AnthropicAgent(BaseAgent):
    """Agent backed by the Anthropic Messages API with tool_use."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required: pip install anthropic"
            ) from exc
        self._model_name = model

    @property
    def framework_label(self) -> str:
        return "Anthropic tool_use"

    async def run_turn(self, user_message: str) -> AgentTurn:
        import anthropic

        client = anthropic.AsyncAnthropic()
        tools  = get_anthropic_schemas()

        # Append user message to history
        self.history.append({"role": "user", "content": user_message})

        all_tool_calls: List[ToolCall] = []
        final_text     = ""

        for iteration in range(self.MAX_ITERATIONS):
            response = await client.messages.create(
                model=self._model_name,
                max_tokens=4096,
                system=self.system_prompt,
                tools=tools,
                messages=self.history,
            )

            # Collect text blocks for final response
            text_blocks = [
                block.text
                for block in response.content
                if block.type == "text"
            ]
            final_text = " ".join(text_blocks).strip()

            # Check for tool_use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks or response.stop_reason == "end_turn":
                # Append assistant message to history
                self.history.append({"role": "assistant", "content": response.content})
                break

            # Append the assistant turn (with tool_use blocks) to history
            self.history.append({"role": "assistant", "content": response.content})

            # Execute all tool calls in this round
            tool_result_blocks: List[Dict[str, Any]] = []
            for block in tool_use_blocks:
                tc = ToolCall(id=block.id, name=block.name, args=block.input or {})
                try:
                    tc.result = await execute_tool(block.name, block.input or {})
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", block.name, exc)
                    tc.result = {"error": str(exc)}

                all_tool_calls.append(tc)
                tool_result_blocks.append(
                    {
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     json.dumps(tc.result, default=str),
                    }
                )

            # Append tool results as a user message (Anthropic protocol)
            self.history.append({"role": "user", "content": tool_result_blocks})

        else:
            logger.warning("AnthropicAgent: reached max iterations (%d)", self.MAX_ITERATIONS)

        return AgentTurn(response_text=final_text, tool_calls=all_tool_calls)
