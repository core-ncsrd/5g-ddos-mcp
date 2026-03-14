"""5G-DDoS agent package.

Provides autonomous agentic loops that let an LLM call the same
tools exposed by the MCP server, via three alternative backends:

  • Anthropic tool_use  (no extra framework)
  • OpenAI function-calling  (also works with Ollama's OpenAI-compat endpoint)
  • LangGraph ReAct agent  (requires: langchain, langchain-core, langgraph)

Usage::

    from src.agents.factory import get_agent
    agent = get_agent(framework="auto")   # or "anthropic" / "openai" / "langgraph"
    agent.run_repl()
"""

from .factory import get_agent

__all__ = ["get_agent"]
