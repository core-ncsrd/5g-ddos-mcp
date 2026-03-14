"""LangGraph ReAct agent.

Uses LangChain's ``create_react_agent`` (from ``langgraph``) with the four
5G-DDoS tools registered via ``src/agents/registry.get_langchain_tools()``.

Supports any LangChain-compatible chat model:
  • ChatAnthropic  (langchain-anthropic)
  • ChatOpenAI     (langchain-openai)
  • ChatOllama     (langchain-ollama / langchain-community)

Required packages::

    pip install langchain langchain-core langgraph
    # and one of:
    pip install langchain-anthropic
    pip install langchain-openai
    pip install langchain-ollama        # or langchain-community

LangGraph ReAct loop:
  User → LLM → [ToolNode] → LLM → … → final response
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import AgentTurn, BaseAgent, ToolCall

logger = logging.getLogger(__name__)


def _get_langchain_model(backend: str, model: Optional[str] = None):
    """Return an appropriate LangChain chat model for the given backend."""
    if backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("pip install langchain-anthropic") from exc
        return ChatAnthropic(model=model or "claude-sonnet-4-6", temperature=0)   # type: ignore[call-arg]

    if backend in ("openai", "openai_compatible"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("pip install langchain-openai") from exc
        kwargs: Dict[str, Any] = {"model": model or "gpt-4o", "temperature": 0}
        if backend == "openai_compatible":
            kwargs["base_url"] = os.getenv("OPENAI_BASE_URL", "")
        return ChatOpenAI(**kwargs)

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama  # type: ignore[no-redef]
            except ImportError as exc:
                raise ImportError(
                    "pip install langchain-ollama  # or langchain-community"
                ) from exc
        return ChatOllama(model=model or "llama3.1", temperature=0)

    raise ValueError(
        f"Unsupported backend for LangGraph agent: {backend!r}. "
        "Choose: anthropic | openai | openai_compatible | ollama"
    )


class LangGraphAgent(BaseAgent):
    """LangGraph ReAct agent wrapping the 5G-DDoS tools."""

    def __init__(self, backend: str = "anthropic", model: Optional[str] = None, **kwargs: Any):
        # Resolve model name before calling super()
        _model = model or {
            "anthropic":        "claude-sonnet-4-6",
            "openai":           "gpt-4o",
            "openai_compatible":"gpt-4o",
            "ollama":           "llama3.1",
        }.get(backend, "unknown")

        super().__init__(model=_model, **kwargs)
        self._backend = backend

        # Lazy-build the compiled graph on first use
        self._graph = None
        self._lc_model = None

    @property
    def framework_label(self) -> str:
        return f"LangGraph ReAct  ({self._backend})"

    def _build_graph(self):
        """Build and compile the LangGraph ReAct agent (lazy, first-turn only)."""
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError as exc:
            raise ImportError("pip install langgraph") from exc

        from langchain_core.messages import SystemMessage
        from .registry import get_langchain_tools

        lc_model = _get_langchain_model(self._backend, self.model)
        tools    = get_langchain_tools()

        self._graph = create_react_agent(
            lc_model,
            tools,
            state_modifier=SystemMessage(content=self.system_prompt),
        )
        self._lc_tools_by_name = {t.name: t for t in tools}

    async def run_turn(self, user_message: str) -> AgentTurn:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        if self._graph is None:
            self._build_graph()

        # Rebuild LangChain message history from our generic history list
        lc_messages = []
        for msg in self.history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            # tool messages are embedded in LangGraph state internally

        lc_messages.append(HumanMessage(content=user_message))
        self.history.append({"role": "user", "content": user_message})

        # Invoke the compiled graph
        result = await self._graph.ainvoke({"messages": lc_messages})  # type: ignore[union-attr]

        # Extract tool calls and final response from the graph's message list
        all_tool_calls: List[ToolCall] = []
        final_text = ""
        tool_results: Dict[str, Any] = {}

        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                tool_results[msg.tool_call_id] = msg.content
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                for raw_tc in msg.tool_calls:
                    tc = ToolCall(
                        id=raw_tc.get("id", ""),
                        name=raw_tc.get("name", ""),
                        args=raw_tc.get("args", {}),
                    )
                    tc.result = tool_results.get(tc.id)
                    all_tool_calls.append(tc)
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)

        self.history.append({"role": "assistant", "content": final_text})
        return AgentTurn(response_text=final_text, tool_calls=all_tool_calls)
