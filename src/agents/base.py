"""Base agent class and shared REPL loop.

Concrete agents (AnthropicAgent, OpenAIAgent, LangGraphAgent) inherit from
``BaseAgent`` and override ``run_turn()``.  The interactive REPL and all Rich
terminal rendering live here so each backend stays focused on its own SDK.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Message types shared across backends
# ──────────────────────────────────────────────────────────────────────────────

class ToolCall:
    """Represents a single tool invocation emitted by the LLM."""
    __slots__ = ("id", "name", "args", "result")

    def __init__(self, id: str, name: str, args: Dict[str, Any]):
        self.id     = id
        self.name   = name
        self.args   = args
        self.result: Optional[Any] = None


class AgentTurn:
    """The complete output of one agent reasoning cycle."""
    __slots__ = ("response_text", "tool_calls")

    def __init__(self, response_text: str = "", tool_calls: Optional[List[ToolCall]] = None):
        self.response_text = response_text
        self.tool_calls    = tool_calls or []


# ──────────────────────────────────────────────────────────────────────────────
# BaseAgent
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a 5G network security analyst with autonomous tool access.
You can detect DDoS attacks, explain incidents, recommend mitigations, and
query historical incident data — all grounded in the NCSRD-DS-5GDDoS dataset
(physical 3GPP 5G testbed, 9 UEs, 5 attack types).

Always call tools when you need concrete data. Think step by step:
  1. If the user describes network conditions → call detect_anomaly first.
  2. If an attack is detected → call explain_attack for a full report.
  3. If the user asks for mitigations → call recommend_response.
  4. For historical patterns → call query_history.

Be concise in your final answers. Present tool results in plain language.
"""


class BaseAgent(ABC):
    """Abstract base for all agent backends."""

    MAX_ITERATIONS = 8   # max tool-call rounds per turn

    def __init__(
        self,
        model: str,
        system_prompt: str = SYSTEM_PROMPT,
        verbose: bool = False,
    ):
        self.model         = model
        self.system_prompt = system_prompt
        self.verbose       = verbose
        self.history: List[Dict[str, Any]] = []   # conversation history

    # ── Subclass contract ─────────────────────────────────────────────────────

    @abstractmethod
    async def run_turn(self, user_message: str) -> AgentTurn:
        """Process one user message, calling tools as needed.

        Implementations must:
        1. Append the user message to ``self.history``.
        2. Loop: send history to LLM → if tool_use blocks, execute them, append
           tool results, repeat.
        3. Return an ``AgentTurn`` with the final text response and all tool
           calls (with their results attached).
        4. Append the final assistant message to ``self.history``.
        """

    @property
    @abstractmethod
    def framework_label(self) -> str:
        """Short label shown in the REPL banner, e.g. 'Anthropic tool_use'."""

    # ── REPL ─────────────────────────────────────────────────────────────────

    def run_repl(self) -> None:
        """Blocking interactive REPL.  Call from a sync context (e.g. CLI)."""
        asyncio.get_event_loop().run_until_complete(self._repl_async())

    async def _repl_async(self) -> None:
        self._print_banner()
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye![/dim]")
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in {"exit", "quit", "bye", "/exit", "/quit"}:
                console.print("[dim]Session ended.[/dim]")
                break
            if stripped.lower() in {"/clear", "clear"}:
                self.history.clear()
                console.print("[dim]Context cleared.[/dim]")
                continue
            if stripped.lower() in {"/tools", "tools"}:
                self._print_tool_table()
                continue

            try:
                turn = await run_turn_with_spinner(self, stripped)
            except Exception as exc:
                logger.exception("Agent error")
                console.print(f"[red]Error:[/red] {exc}")
                continue

            self._render_turn(turn)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        console.print()
        console.print(
            Panel.fit(
                f"[bold blue]5G-DDoS Agent[/bold blue]  ·  "
                f"[dim]{self.framework_label}[/dim]  ·  "
                f"model [green]{self.model}[/green]\n"
                "[dim]Type your question, paste telemetry, or describe a network event.\n"
                "Commands: /clear  /tools  exit[/dim]",
                border_style="blue",
            )
        )

    def _print_tool_table(self) -> None:
        table = Table(title="Available Tools", box=box.ROUNDED, border_style="blue")
        table.add_column("Tool", style="cyan")
        table.add_column("Description")
        from .registry import TOOL_SCHEMAS_OPENAI
        for t in TOOL_SCHEMAS_OPENAI:
            fn = t["function"]
            table.add_row(fn["name"], fn["description"][:90] + "…")
        console.print(table)

    def _render_turn(self, turn: AgentTurn) -> None:
        """Print tool calls (if verbose or always) and the final LLM response."""
        if turn.tool_calls:
            for tc in turn.tool_calls:
                self._render_tool_call(tc)

        if turn.response_text.strip():
            console.print(
                Panel(
                    Markdown(turn.response_text),
                    title="[bold green]Agent[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                )
            )

    def _render_tool_call(self, tc: ToolCall) -> None:
        args_str = json.dumps(tc.args, indent=2, default=str)
        console.print(
            Panel(
                Syntax(args_str, "json", theme="monokai", word_wrap=True),
                title=f"[bold yellow]⚙ Tool call:[/bold yellow] [cyan]{tc.name}[/cyan]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        if tc.result is not None:
            result_str = (
                json.dumps(tc.result, indent=2, default=str)
                if isinstance(tc.result, (dict, list))
                else str(tc.result)
            )
            # Truncate very long tool results for display
            if len(result_str) > 1200:
                result_str = result_str[:1200] + "\n… (truncated)"
            console.print(
                Panel(
                    Syntax(result_str, "json", theme="monokai", word_wrap=True),
                    title=f"[bold magenta]↩ Result:[/bold magenta] [cyan]{tc.name}[/cyan]",
                    border_style="magenta",
                    padding=(0, 1),
                )
            )


# ──────────────────────────────────────────────────────────────────────────────
# Spinner helper (non-blocking)
# ──────────────────────────────────────────────────────────────────────────────

async def run_turn_with_spinner(agent: BaseAgent, message: str) -> AgentTurn:
    """Run ``agent.run_turn()`` while showing a Rich status spinner."""
    with console.status("[dim]Thinking…[/dim]", spinner="dots"):
        return await agent.run_turn(message)
