"""
5G-DDoS CLI
===========
Command-line interface for the 5G-DDoS MCP server tools.

Usage:
    5g-ddos --help
    5g-ddos server
    5g-ddos detect --input telemetry.csv
    5g-ddos detect --ul-bitrate 60000000 --gnb-cpu-load 94
    5g-ddos explain --attack-type gtp_u_flood --severity critical
    5g-ddos recommend --attack-type syn_flood --severity high --slice eMBB
    5g-ddos history [--attack-type gtp_u_flood] [--limit 10]
    5g-ddos dataset status
    5g-ddos dataset download
    5g-ddos train [--nrows N]
    5g-ddos version
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app     = typer.Typer(
    name            = "5g-ddos",
    help            = "5G DDoS detection and response CLI · MCP server · XGBoost · NCSRD dataset",
    add_completion  = True,
    rich_markup_mode= "rich",
    no_args_is_help = True,
)
console = Console()

# Sub-app groups
dataset_app = typer.Typer(help="Manage the NCSRD-DS-5GDDoS dataset.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity_color(severity: str) -> str:
    return {
        "critical":    "bold red",
        "high":        "red",
        "medium-high": "yellow",
        "medium":      "yellow",
        "none":        "green",
    }.get(severity.lower(), "white")


def _print_json(data: dict) -> None:
    syntax = Syntax(json.dumps(data, indent=2, default=str), "json", theme="monokai", line_numbers=False)
    console.print(syntax)


def _run(coro):
    """Run an async coroutine from sync CLI context."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# server
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def server(
    backend: Optional[str] = typer.Option(None, "--backend", "-b",
        help="LLM backend override: claude | ollama | openai_compatible"),
    model: Optional[str] = typer.Option(None, "--model", "-m",
        help="Model name override (e.g. llama3.2, claude-sonnet-4-6)"),
):
    """
    Start the MCP server.

    The server exposes four tools to any MCP-compatible LLM client:
    detect_anomaly, explain_attack, recommend_response, query_history.
    """
    if backend:
        os.environ["LLM_BACKEND"] = backend
    if model:
        b = os.environ.get("LLM_BACKEND", "claude")
        key = {"claude": "CLAUDE_MODEL", "ollama": "OLLAMA_MODEL"}.get(b, "OPENAI_MODEL")
        os.environ[key] = model

    from src.config import server_config
    b = os.environ.get("LLM_BACKEND", "claude")

    console.print(Panel.fit(
        f"[bold cyan]5G-DDoS MCP Server[/bold cyan]\n"
        f"LLM backend : [yellow]{b}[/yellow]\n"
        f"Tools       : detect_anomaly · explain_attack · recommend_response · query_history",
        border_style="cyan",
    ))

    from src.server import mcp
    mcp.run()


# ─────────────────────────────────────────────────────────────────────────────
# agent  (interactive REPL)
# ─────────────────────────────────────────────────────────────────────────────

FRAMEWORK_HELP = (
    "Agent framework: auto | anthropic | openai | ollama | openai_compatible "
    "| langchain | langgraph.  "
    "'auto' picks from LLM_BACKEND env var (default: anthropic)."
)

@app.command()
def agent(
    framework: str = typer.Option(
        "auto", "--framework", "-f", help=FRAMEWORK_HELP
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help="Override LLM_BACKEND env var (anthropic | openai | ollama | openai_compatible).",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name override (e.g. claude-sonnet-4-6, llama3.1, gpt-4o).",
    ),
    system: Optional[str] = typer.Option(
        None, "--system",
        help="Custom system prompt (replaces the default analyst persona).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show raw tool-call JSON even when not in an error path.",
    ),
):
    """
    Start an interactive agent REPL with autonomous tool use.

    The agent can call detect_anomaly, explain_attack, recommend_response, and
    query_history on its own, reasoning step-by-step until it has a final answer.

    \b
    Framework options
    -----------------
    auto          Uses LLM_BACKEND env var to pick (default: anthropic)
    anthropic     Raw Anthropic tool_use API — no extra packages
    openai        OpenAI function-calling — requires: openai
    ollama        Ollama /v1/chat/completions — requires: openai
    langgraph     LangGraph ReAct agent — requires: langchain langgraph

    \b
    Examples
    --------
    5g-ddos agent
    5g-ddos agent --framework ollama --model phi4
    5g-ddos agent --framework langgraph --backend anthropic
    5g-ddos agent --framework openai --model gpt-4o --verbose
    """
    if backend:
        os.environ["LLM_BACKEND"] = backend

    from src.agents.factory import get_agent

    try:
        ag = get_agent(
            framework=framework,
            model=model,
            system_prompt=system,
            verbose=verbose,
        )
    except ImportError as exc:
        console.print(f"[red]Missing dependency:[/red] {exc}")
        raise typer.Exit(1)
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    ag.run_repl()


# ─────────────────────────────────────────────────────────────────────────────
# detect
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def detect(
    # ── file input ──────────────────────────────────────────────────────────
    input: Optional[Path] = typer.Option(None, "--input", "-i",
        help="CSV file of telemetry rows to classify."),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
        help="Write results to this JSON file."),
    format: str = typer.Option("table", "--format", "-f",
        help="Output format: table | json | csv"),
    # ── inline single-snapshot args ─────────────────────────────────────────
    ul_bitrate:                 float = typer.Option(0.0,  help="UE uplink bitrate (bps)"),
    dl_bitrate:                 float = typer.Option(0.0,  help="UE downlink bitrate (bps)"),
    ul_retx:                    float = typer.Option(0.0,  help="UL retransmission count"),
    dl_retx:                    float = typer.Option(0.0,  help="DL retransmission count"),
    ul_mcs:                     float = typer.Option(0.0,  help="UL modulation coding scheme"),
    dl_mcs:                     float = typer.Option(0.0,  help="DL modulation coding scheme"),
    ul_phr:                     float = typer.Option(0.0,  help="UL power headroom"),
    cell_id:                    int   = typer.Option(0,    help="Cell ID (1, 2, or 3)"),
    pucch_snr:                  float = typer.Option(0.0,  help="PUCCH SNR (dB)"),
    pusch_snr:                  float = typer.Option(0.0,  help="PUSCH SNR (dB)"),
    enb_dl_bitrate:             float = typer.Option(0.0,  help="eNB DL bitrate"),
    enb_ul_bitrate:             float = typer.Option(0.0,  help="eNB UL bitrate"),
    enb_dl_usage_ratio:         float = typer.Option(0.0,  help="eNB DL usage ratio (0-1)"),
    enb_ul_usage_ratio:         float = typer.Option(0.0,  help="eNB UL usage ratio (0-1)"),
    gnb_cpu_load:               float = typer.Option(0.0,  help="gNB CPU load (%%)"),
    pdu_session_setup_request:  float = typer.Option(0.0,  help="MME PDU session setup requests"),
    pdu_session_setup_response: float = typer.Option(0.0,  help="MME PDU session setup responses"),
    paging_attempts:            float = typer.Option(0.0,  help="MME paging attempts"),
    context_setup_request:      float = typer.Option(0.0,  help="MME context setup requests"),
    context_setup_response:     float = typer.Option(0.0,  help="MME context setup responses"),
    abnormal_release_count:     float = typer.Option(0.0,  help="MME abnormal release count"),
    ue_id:                      Optional[str] = typer.Option(None, help="UE identifier"),
):
    """
    Detect 5G DDoS attacks from telemetry.

    Accepts either a [bold]CSV file[/bold] (--input) for batch classification
    or inline metric flags for a single snapshot.

    Examples:

      [dim]# Batch from CSV[/dim]
      5g-ddos detect --input telemetry.csv --output report.json

      [dim]# Single snapshot[/dim]
      5g-ddos detect --ul-bitrate 60000000 --gnb-cpu-load 94 --ul-retx 800

      [dim]# JSON output[/dim]
      5g-ddos detect --input telemetry.csv --format json
    """
    from src.tools.detect_anomaly import detect_anomaly as _detect

    if input:
        _detect_csv(input, output, format)
        return

    # ── Single snapshot ───────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("[cyan]Running detection…"), transient=True) as p:
        p.add_task("")
        result = _run(_detect(
            ul_bitrate=ul_bitrate, dl_bitrate=dl_bitrate,
            ul_retx=ul_retx, dl_retx=dl_retx,
            ul_mcs=ul_mcs, dl_mcs=dl_mcs, ul_phr=ul_phr,
            cell_id=cell_id, pucch_snr=pucch_snr, pusch_snr=pusch_snr,
            enb_dl_bitrate=enb_dl_bitrate, enb_ul_bitrate=enb_ul_bitrate,
            enb_dl_usage_ratio=enb_dl_usage_ratio,
            enb_ul_usage_ratio=enb_ul_usage_ratio,
            gnb_cpu_load=gnb_cpu_load,
            pdu_session_setup_request=pdu_session_setup_request,
            pdu_session_setup_response=pdu_session_setup_response,
            paging_attempts=paging_attempts,
            context_setup_request=context_setup_request,
            context_setup_response=context_setup_response,
            abnormal_release_count=abnormal_release_count,
            ue_id=ue_id,
        ))

    _print_detection_result(result, format)

    if output:
        output.write_text(json.dumps(result, indent=2, default=str))
        console.print(f"\n[dim]Saved to {output}[/dim]")


def _detect_csv(input_path: Path, output: Optional[Path], fmt: str) -> None:
    """Run batch detection over a CSV file."""
    try:
        import pandas as pd
    except ImportError:
        console.print("[red]pandas is required: pip install pandas[/red]")
        raise typer.Exit(1)

    from src.data.preprocessor import normalise_columns, extract_features
    from src.models.classifier import get_classifier
    from src.config import ALL_FEATURES

    console.print(f"[cyan]Loading[/cyan] {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    df = normalise_columns(df)

    clf = get_classifier()
    results = []

    with Progress(SpinnerColumn(), TextColumn("[cyan]Classifying rows…"), transient=True) as p:
        p.add_task("")
        for _, row in df.iterrows():
            features = {f: float(row.get(f, 0.0)) for f in ALL_FEATURES}
            try:
                r = clf.predict(features)
            except Exception:
                from src.tools.detect_anomaly import _heuristic_detection
                r = _heuristic_detection(features)
            results.append(r)

    attacks   = [r for r in results if r["is_attack"]]
    by_type:dict = {}
    for r in attacks:
        t = r["attack_type"]
        by_type[t] = by_type.get(t, 0) + 1

    console.print(f"\n[bold]Batch Detection Results[/bold] — {len(df)} rows")
    console.print(f"  Attacks detected : [red]{len(attacks)}[/red] / {len(df)}")
    console.print(f"  Benign           : [green]{len(df)-len(attacks)}[/green]")
    if by_type:
        console.print(f"  By type          : {by_type}")

    if fmt == "table":
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("#",          style="dim",  width=6)
        table.add_column("Attack",     style="bold", width=14)
        table.add_column("Confidence", width=11)
        table.add_column("Severity",   width=12)
        for i, r in enumerate(results):
            sev   = r.get("severity", "")
            color = _severity_color(sev)
            table.add_row(
                str(i),
                r["attack_type"],
                f"{r['binary_probability']:.1%}",
                Text(sev, style=color),
            )
        console.print(table)

    if output:
        output.write_text(json.dumps(results, indent=2, default=str))
        console.print(f"\n[dim]Saved {len(results)} results to {output}[/dim]")


def _print_detection_result(result: dict, fmt: str) -> None:
    if fmt == "json":
        _print_json(result)
        return

    is_attack = result.get("is_attack", False)
    attack    = result.get("attack_type", "benign")
    prob      = result.get("binary_probability", 0.0)
    sev       = result.get("severity", "none")
    color     = _severity_color(sev)

    status = "[bold red]⚠  ATTACK DETECTED[/bold red]" if is_attack else "[bold green]✓  BENIGN[/bold green]"
    console.print(Panel(
        f"{status}\n\n"
        f"  Type       : [bold]{attack.replace('_', ' ').upper()}[/bold]\n"
        f"  Confidence : [cyan]{prob:.1%}[/cyan]\n"
        f"  Severity   : [{color}]{sev.upper()}[/{color}]",
        border_style="red" if is_attack else "green",
        title="Detection Result",
    ))

    if fmt == "table":
        probs = result.get("class_probabilities", {})
        if probs:
            table = Table("Attack Type", "Probability", box=box.SIMPLE, show_header=True,
                          header_style="bold dim")
            for label, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                bar   = "█" * int(p * 20)
                color = _severity_color(label) if p > 0.05 else "dim"
                table.add_row(label, f"[{color}]{p:.4f}  {bar}[/{color}]")
            console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# explain
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def explain(
    attack_type: str  = typer.Option(..., "--attack-type", "-a",
        help="Attack type: syn_flood | udp_flood | icmp_flood | dns_flood | gtp_u_flood"),
    severity:    str  = typer.Option("high", "--severity", "-s",
        help="Severity: none | medium | medium-high | high | critical"),
    ue_id:       Optional[str] = typer.Option(None,  "--ue-id",     help="UE identifier"),
    imeisv:      Optional[str] = typer.Option(None,  "--imeisv",    help="UE IMEISV"),
    ue_ip:       Optional[str] = typer.Option(None,  "--ue-ip",     help="UE IP address"),
    cells:       Optional[str] = typer.Option(None,  "--cells",     help="Affected cells, e.g. '1,3'"),
    slice_type:  Optional[str] = typer.Option(None,  "--slice",     help="Slice type: eMBB | URLLC | mMTC"),
    timestamp:   Optional[str] = typer.Option(None,  "--timestamp", help="ISO 8601 timestamp"),
    no_remediation: bool       = typer.Option(False, "--no-remediation", help="Omit remediation section"),
    output: Optional[Path]     = typer.Option(None,  "--output", "-o", help="Save report to file"),
    format: str                = typer.Option("text", "--format", "-f", help="Output format: text | json"),
):
    """
    Generate an LLM-powered incident report for a detected attack.

    Examples:

      5g-ddos explain --attack-type gtp_u_flood --severity critical --slice eMBB

      5g-ddos explain --attack-type syn_flood --ue-ip 192.168.1.101 \\
              --imeisv 86620003000000 --cells 1,3 --output report.md
    """
    from src.tools.explain_attack import explain_attack as _explain
    from src.config import ATTACK_SEVERITY

    detection = {
        "is_attack":          True,
        "attack_type":        attack_type,
        "severity":           severity,
        "binary_probability": 0.99,
        "class_probabilities": {},
    }

    with Progress(SpinnerColumn(), TextColumn("[cyan]Generating incident report…"), transient=True) as p:
        p.add_task("")
        result = _run(_explain(
            detection_result    = detection,
            ue_identity         = ue_id,
            imeisv              = imeisv,
            ue_ip               = ue_ip,
            affected_cells      = cells,
            slice_type          = slice_type,
            timestamp           = timestamp,
            include_remediation = not no_remediation,
        ))

    if format == "json":
        _print_json(result)
    else:
        sev_color = _severity_color(result.get("severity", ""))
        console.print(Panel(
            f"[bold]{attack_type.replace('_', ' ').upper()}[/bold]  "
            f"[{sev_color}]{result.get('severity','').upper()}[/{sev_color}]  "
            f"→  {result.get('affected_layer', '')}",
            title="[bold cyan]Incident Report[/bold cyan]",
            border_style="cyan",
        ))
        console.print(result.get("report_text", ""))

    if output:
        text = result.get("report_text", json.dumps(result, indent=2))
        output.write_text(text)
        console.print(f"\n[dim]Report saved to {output}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# recommend
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def recommend(
    attack_type:   str           = typer.Option(...,    "--attack-type", "-a",
        help="Attack type: syn_flood | udp_flood | icmp_flood | dns_flood | gtp_u_flood"),
    severity:      str           = typer.Option("high", "--severity", "-s",
        help="Severity: none | medium | medium-high | high | critical"),
    slice_type:    Optional[str] = typer.Option(None,   "--slice",        help="Slice: eMBB | URLLC | mMTC"),
    ue_ip:         Optional[str] = typer.Option(None,   "--ue-ip",        help="Attacking UE IP"),
    imeisv:        Optional[str] = typer.Option(None,   "--imeisv",       help="Attacking UE IMEISV"),
    cells:         Optional[str] = typer.Option(None,   "--cells",        help="Affected cells, e.g. '1,3'"),
    ue_count:      int           = typer.Option(0,      "--ue-count",     help="Total UEs on slice"),
    auto_execute:  bool          = typer.Option(False,  "--auto-execute", help="Auto-execute via RESPONSE_API_URL"),
    output:        Optional[Path]= typer.Option(None,   "--output", "-o", help="Save to JSON file"),
    format:        str           = typer.Option("table","--format", "-f", help="table | json"),
):
    """
    Get LLM-powered mitigation recommendations for a detected attack.

    Examples:

      5g-ddos recommend --attack-type gtp_u_flood --severity critical --slice eMBB

      5g-ddos recommend --attack-type syn_flood --ue-ip 10.0.0.5 \\
              --cells 1,3 --auto-execute
    """
    from src.tools.recommend_response import recommend_response as _recommend

    cell_list = [int(c.strip()) for c in cells.split(",")] if cells else None

    with Progress(SpinnerColumn(), TextColumn("[cyan]Generating recommendations…"), transient=True) as p:
        p.add_task("")
        result = _run(_recommend(
            attack_type        = attack_type,
            severity           = severity,
            slice_type         = slice_type,
            affected_cells     = cell_list,
            imeisv             = imeisv,
            ue_ip              = ue_ip,
            current_ue_count   = ue_count,
            allow_auto_execute = auto_execute,
        ))

    if format == "json":
        _print_json(result)
    else:
        recs = result.get("recommendations", [])
        table = Table(
            "Priority", "Action", "Description", "Disruption", "Effectiveness",
            box=box.ROUNDED, header_style="bold cyan", show_header=True,
        )
        priority_color = {"immediate": "red", "short-term": "yellow", "long-term": "green"}
        for r in recs:
            pri   = r.get("priority", "")
            color = priority_color.get(pri, "white")
            table.add_row(
                Text(pri, style=f"bold {color}"),
                r.get("action", ""),
                r.get("description", ""),
                r.get("disruption", ""),
                r.get("effectiveness", ""),
            )
        console.print(table)

        # API examples
        api_cmds = result.get("api_commands", [])
        if api_cmds:
            console.print("\n[bold]Generic REST API Examples[/bold]")
            for cmd in api_cmds:
                console.print(f"  [cyan]{cmd['method']}[/cyan] {cmd['endpoint']}")
                console.print(f"  [dim]{cmd['payload']}[/dim]\n")

        # LLM rationale
        rationale = result.get("rationale", "")
        if rationale:
            console.print(Panel(rationale, title="[bold]LLM Rationale[/bold]", border_style="dim"))

        # Auto-execute feedback
        executed = result.get("auto_executed", [])
        if executed:
            console.print(f"\n[green]Auto-executed:[/green] {', '.join(executed)}")
        elif auto_execute:
            console.print("[yellow]Auto-execute requested but RESPONSE_API_URL not configured.[/yellow]")

    if output:
        output.write_text(json.dumps(result, indent=2, default=str))
        console.print(f"\n[dim]Saved to {output}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# history
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def history(
    attack_type:   Optional[str] = typer.Option(None,  "--attack-type", "-a", help="Filter by attack type"),
    severity:      Optional[str] = typer.Option(None,  "--severity",    "-s", help="Filter by severity"),
    slice_type:    Optional[str] = typer.Option(None,  "--slice",             help="Filter by slice type"),
    cell_id:       Optional[int] = typer.Option(None,  "--cell",              help="Filter by cell ID"),
    ue_ip:         Optional[str] = typer.Option(None,  "--ue-ip",             help="Filter by UE IP"),
    imeisv:        Optional[str] = typer.Option(None,  "--imeisv",            help="Filter by IMEISV"),
    limit:         int           = typer.Option(10,    "--limit",       "-n", help="Max results to show"),
    hours:         Optional[int] = typer.Option(None,  "--hours",             help="Only last N hours"),
    include_benign:bool          = typer.Option(False, "--include-benign",    help="Include benign records"),
    format:        str           = typer.Option("table","--format", "-f",     help="table | json"),
):
    """
    Search the historical incident database.

    The database is pre-seeded with reference patterns from the NCSRD-DS-5GDDoS
    dataset and grows with each detected incident.

    Examples:

      5g-ddos history

      5g-ddos history --attack-type gtp_u_flood --slice eMBB

      5g-ddos history --hours 24 --format json
    """
    from src.tools.query_history import query_history as _query

    with Progress(SpinnerColumn(), TextColumn("[cyan]Querying history…"), transient=True) as p:
        p.add_task("")
        result = _run(_query(
            attack_type      = attack_type,
            severity         = severity,
            slice_type       = slice_type,
            cell_id          = cell_id,
            ue_ip            = ue_ip,
            imeisv           = imeisv,
            limit            = limit,
            include_benign   = include_benign,
            time_range_hours = hours,
        ))

    if format == "json":
        _print_json(result)
        return

    console.print(f"\n[bold]Incident History[/bold]  [dim]{result.get('summary', '')}[/dim]\n")

    incidents = result.get("incidents", [])
    if not incidents:
        console.print("[dim]No incidents found.[/dim]")
        return

    table = Table(
        "ID", "Attack Type", "Severity", "Slice", "Cell", "UE IP", "Confidence",
        box=box.ROUNDED, header_style="bold cyan",
    )
    for inc in incidents:
        sev   = inc.get("severity", "")
        color = _severity_color(sev)
        table.add_row(
            str(inc.get("id", ""))[:12],
            inc.get("attack_type", ""),
            Text(sev, style=color),
            inc.get("slice_type") or "—",
            str(inc.get("cell_id") or "—"),
            inc.get("ue_ip") or "—",
            f"{inc.get('confidence', 0):.0%}" if inc.get("confidence") else "—",
        )
    console.print(table)

    freq = result.get("attack_frequency", {})
    if freq:
        console.print("\n[bold]Attack Frequency (all time)[/bold]")
        for atype, count in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            bar   = "█" * min(count, 40)
            color = _severity_color(atype)
            console.print(f"  [{color}]{atype:<18}[/{color}]  {bar}  {count}")


# ─────────────────────────────────────────────────────────────────────────────
# train
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def train(
    nrows:      Optional[int]  = typer.Option(None,  "--nrows",      "-n", help="Limit rows loaded (for quick tests)"),
    no_balance: bool           = typer.Option(False, "--no-balance",       help="Skip SMOTE-Tomek class balancing"),
    test_frac:  float          = typer.Option(0.2,   "--test-frac",        help="Test set fraction (default 0.2)"),
):
    """
    Train the XGBoost classifier on the NCSRD-DS-5GDDoS dataset.

    Produces:
      models/xgb_binary_classifier.pkl
      models/xgb_multiclass_classifier.pkl
      models/feature_scaler.pkl
      models/training_report.json

    The dataset must be downloaded first:

      5g-ddos dataset download

    Examples:

      5g-ddos train                     [dim]# full dataset[/dim]
      5g-ddos train --nrows 100000      [dim]# quick smoke-test[/dim]
      5g-ddos train --no-balance        [dim]# skip SMOTE-Tomek[/dim]
    """
    from src.data.loader import check_dataset_available
    status = check_dataset_available()
    merged = status.get("merged", {})

    if not merged.get("exists"):
        console.print(Panel(
            "[red]Merged dataset file not found.[/red]\n\n"
            "Download it first:\n"
            "  [cyan]5g-ddos dataset download[/cyan]",
            border_style="red",
        ))
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Training 5G-DDoS Classifier[/bold cyan]\n"
        f"Dataset : {merged['path']}\n"
        f"Rows    : {'all' if not nrows else nrows:,}\n"
        f"Balance : {'SMOTE-Tomek' if not no_balance else 'none'}",
        border_style="cyan",
    ))

    # Delegate to the training script
    import subprocess
    cmd = [sys.executable, "scripts/train_model.py"]
    if nrows:
        cmd += ["--nrows", str(nrows)]
    if no_balance:
        cmd += ["--no-balance"]
    cmd += ["--test-frac", str(test_frac)]

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    raise typer.Exit(result.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# dataset status
# ─────────────────────────────────────────────────────────────────────────────

@dataset_app.command("status")
def dataset_status():
    """Show which dataset files are present and their sizes."""
    from src.data.loader import check_dataset_available
    status = check_dataset_available()

    table = Table("Key", "Filename", "Present", "Size", box=box.ROUNDED, header_style="bold cyan")
    total_mb = 0.0
    for key, info in status.items():
        present  = info["exists"]
        size_mb  = info["size_mb"]
        if size_mb:
            total_mb += size_mb
        table.add_row(
            key,
            info["filename"],
            "[green]✓[/green]" if present else "[red]✗[/red]",
            f"{size_mb} MB" if size_mb else "—",
        )

    console.print("\n[bold]NCSRD-DS-5GDDoS Dataset Status[/bold]")
    console.print(table)

    downloaded = sum(1 for v in status.values() if v["exists"])
    total      = len(status)
    console.print(f"\n  {downloaded}/{total} files present · {total_mb:.0f} MB on disk")

    if downloaded < total:
        console.print(
            "\n[yellow]Some files are missing.[/yellow] "
            "Run [cyan]5g-ddos dataset download[/cyan] to get them."
        )
    else:
        console.print("\n[green]All files present.[/green] Run [cyan]5g-ddos train[/cyan] to build the model.")


# ─────────────────────────────────────────────────────────────────────────────
# dataset download
# ─────────────────────────────────────────────────────────────────────────────

@dataset_app.command("download")
def dataset_download(
    auto: bool = typer.Option(False, "--auto", help="Run the download script automatically"),
):
    """
    Show dataset download instructions (or trigger automatic download).

    The NCSRD-DS-5GDDoS dataset (~620 MB) is hosted on Zenodo under CC BY 4.0.
    DOI: https://doi.org/10.5281/zenodo.13900057

    Examples:

      5g-ddos dataset download           [dim]# show instructions[/dim]
      5g-ddos dataset download --auto    [dim]# run download script[/dim]
    """
    if auto:
        import subprocess
        script = Path(__file__).parent.parent / "scripts" / "download_dataset.sh"
        if not script.exists():
            console.print("[red]download_dataset.sh not found.[/red]")
            raise typer.Exit(1)
        result = subprocess.run(["bash", str(script)])
        raise typer.Exit(result.returncode)

    console.print(Panel(
        "[bold cyan]NCSRD-DS-5GDDoS Dataset[/bold cyan]\n\n"
        "[bold]DOI :[/bold]     https://doi.org/10.5281/zenodo.13900057\n"
        "[bold]URL :[/bold]     https://zenodo.org/records/13900057\n"
        "[bold]Size:[/bold]     ~620 MB (7 CSV/XLSX files)\n"
        "[bold]License:[/bold]  CC BY 4.0\n\n"
        "[bold]Option 1 — Automatic[/bold]\n"
        "  [cyan]5g-ddos dataset download --auto[/cyan]\n"
        "  or: [cyan]bash scripts/download_dataset.sh[/cyan]\n\n"
        "[bold]Option 2 — Manual[/bold]\n"
        "  1. Go to https://zenodo.org/records/13900057\n"
        "  2. Download all 7 files\n"
        "  3. Place them in the [cyan]./data/[/cyan] directory\n\n"
        "[bold]Then train the model:[/bold]\n"
        "  [cyan]5g-ddos train[/cyan]",
        border_style="cyan",
        title="Dataset Download Instructions",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# version
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def version():
    """Show version and environment info."""
    from src.config import llm_config, server_config
    from src.data.loader import check_dataset_available
    from src.config import MODEL_ARTIFACT_BINARY

    ds     = check_dataset_available()
    ds_ok  = sum(1 for v in ds.values() if v["exists"])
    model_ok = MODEL_ARTIFACT_BINARY.exists()
    backend  = os.environ.get("LLM_BACKEND", llm_config.backend)

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style="bold dim", width=22)
    table.add_column()

    table.add_row("Version",         "1.0.0")
    table.add_row("Python",          sys.version.split()[0])
    table.add_row("LLM backend",     backend)
    table.add_row("Dataset files",   f"{ds_ok}/7 present")
    table.add_row("ML model",        "[green]trained[/green]" if model_ok else "[yellow]not trained[/yellow]")
    table.add_row("MCP port",        str(server_config.port))
    table.add_row("Threshold",       str(server_config.anomaly_threshold))
    table.add_row("Dataset DOI",     "10.5281/zenodo.13900057")

    console.print(Panel(table, title="[bold cyan]5G-DDoS MCP[/bold cyan]", border_style="cyan"))


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
