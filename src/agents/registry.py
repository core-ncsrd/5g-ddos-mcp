"""Central tool registry for 5G-DDoS agents.

Exposes the four MCP tools in three formats:

  1. ``TOOL_SCHEMAS_OPENAI``   – list[dict] ready for the ``tools=`` param of
     any OpenAI-compatible API (also used by Ollama's /v1/chat/completions).

  2. ``get_anthropic_schemas()`` – same schemas converted to Anthropic's
     ``tool_use`` format (``input_schema`` key instead of ``parameters``).

  3. ``get_langchain_tools()``  – list of ``StructuredTool`` objects for use
     inside a LangGraph ReAct agent.  Requires: ``langchain``, ``langchain-core``.

All tool calls are dispatched through ``execute_tool(name, args)`` which
bridges the simplified agent-facing parameter set to the actual async
implementations in ``src/tools/``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic input models
# Used by LangChain StructuredTool (args_schema) and for IDE auto-complete.
# ──────────────────────────────────────────────────────────────────────────────

class DetectAnomalyInput(BaseModel):
    """Telemetry snapshot for anomaly detection."""
    ul_bitrate:                float = Field(0.0,  description="UE uplink bitrate in bps")
    dl_bitrate:                float = Field(0.0,  description="UE downlink bitrate in bps")
    ul_retx:                   float = Field(0.0,  description="UL retransmission ratio 0–1")
    dl_retx:                   float = Field(0.0,  description="DL retransmission ratio 0–1")
    ul_mcs:                    float = Field(0.0,  description="Uplink Modulation & Coding Scheme")
    dl_mcs:                    float = Field(0.0,  description="Downlink Modulation & Coding Scheme")
    ul_phr:                    float = Field(0.0,  description="UL Power Headroom")
    cell_id:                   int   = Field(0,    description="Serving cell ID (1–3)")
    pucch_snr:                 float = Field(0.0,  description="PUCCH signal-to-noise ratio (dB)")
    pusch_snr:                 float = Field(0.0,  description="PUSCH signal-to-noise ratio (dB)")
    enb_dl_bitrate:            float = Field(0.0,  description="eNB aggregate DL bitrate (bps)")
    enb_ul_bitrate:            float = Field(0.0,  description="eNB aggregate UL bitrate (bps)")
    enb_dl_usage_ratio:        float = Field(0.0,  description="eNB DL resource usage 0–1")
    enb_ul_usage_ratio:        float = Field(0.0,  description="eNB UL resource usage 0–1")
    gnb_cpu_load:              float = Field(0.0,  description="gNB CPU load 0–100")
    pdu_session_setup_request: float = Field(0.0,  description="PDU session setup requests/s")
    pdu_session_setup_response:float = Field(0.0,  description="PDU session setup responses/s")
    paging_attempts:           float = Field(0.0,  description="Paging attempts/s")
    context_setup_request:     float = Field(0.0,  description="S1AP context setup requests/s")
    context_setup_response:    float = Field(0.0,  description="S1AP context setup responses/s")
    abnormal_release_count:    float = Field(0.0,  description="Abnormal bearer releases/s")
    ue_id:                     Optional[str] = Field(None, description="UE identifier, e.g. 'UE-001'")
    timestamp:                 Optional[str] = Field(None, description="ISO 8601 timestamp")


class ExplainAttackInput(BaseModel):
    """Parameters for generating an incident report."""
    attack_type:       str            = Field(...,  description="syn_flood | udp_flood | icmp_flood | dns_flood | gtp_u_flood")
    severity:          str            = Field(...,  description="low | medium | high | critical")
    slice_type:        Optional[str]  = Field(None, description="eMBB | URLLC | mMTC")
    ue_ip:             Optional[str]  = Field(None, description="Attacking UE IP address")
    imeisv:            Optional[str]  = Field(None, description="IMEISV of attacking UE")
    affected_cells:    Optional[str]  = Field(None, description="Comma-separated cell IDs, e.g. '1,2'")
    binary_probability:Optional[float]= Field(None, description="ML attack confidence 0–1")
    ue_id:             Optional[str]  = Field(None, description="UE identifier")


class RecommendResponseInput(BaseModel):
    """Parameters for generating mitigation recommendations."""
    attack_type:   str           = Field(...,  description="syn_flood | udp_flood | icmp_flood | dns_flood | gtp_u_flood")
    severity:      str           = Field(...,  description="low | medium | high | critical")
    slice_type:    Optional[str] = Field(None, description="eMBB | URLLC | mMTC")
    ue_ip:         Optional[str] = Field(None, description="Attacking UE IP address")
    imeisv:        Optional[str] = Field(None, description="IMEISV of attacking UE")
    affected_cells:Optional[str] = Field(None, description="Comma-separated cell IDs")
    auto_execute:  bool          = Field(False, description="Execute low-risk actions via RESPONSE_API_URL if set")


class QueryHistoryInput(BaseModel):
    """Filters for querying the incident history database."""
    attack_type:      Optional[str] = Field(None, description="Filter by attack type")
    severity:         Optional[str] = Field(None, description="Filter by severity")
    slice_type:       Optional[str] = Field(None, description="Filter by network slice")
    cell_id:          Optional[int] = Field(None, description="Filter by cell ID (1–3)")
    ue_ip:            Optional[str] = Field(None, description="Filter by attacking UE IP")
    imeisv:           Optional[str] = Field(None, description="Filter by IMEISV")
    time_range_hours: int           = Field(24,   description="Look-back window in hours")
    limit:            int           = Field(10,   description="Maximum incidents to return")


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible JSON schemas
# ──────────────────────────────────────────────────────────────────────────────

def _model_to_openai_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to an OpenAI-compatible JSON schema."""
    schema = model.model_json_schema()
    # Remove pydantic $defs/$ref noise – keep flat for LLM clarity
    schema.pop("title", None)
    return schema


TOOL_SCHEMAS_OPENAI: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "detect_anomaly",
            "description": (
                "Classify a real-time 5G telemetry snapshot as benign or a specific DDoS attack "
                "(SYN flood, UDP flood, ICMP flood, DNS flood, GTP-U flood) using the trained "
                "XGBoost classifier or rule-based heuristics. Provide as many telemetry metrics "
                "as you have available; all parameters are optional."
            ),
            "parameters": _model_to_openai_schema(DetectAnomalyInput),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_attack",
            "description": (
                "Generate a detailed LLM-powered incident report for a detected 5G DDoS attack. "
                "Analyses the attack against the NCSRD-DS-5GDDoS dataset knowledge to explain "
                "which 5G protocol layers are affected and which telemetry metrics were decisive."
            ),
            "parameters": _model_to_openai_schema(ExplainAttackInput),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_response",
            "description": (
                "Recommend proportional mitigation actions for a 5G DDoS attack: slice isolation, "
                "UE blacklisting, rate limiting, traffic rerouting. Returns LLM rationale plus "
                "example generic REST API payloads adaptable to any NMS."
            ),
            "parameters": _model_to_openai_schema(RecommendResponseInput),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_history",
            "description": (
                "Search the historical incident database for past 5G DDoS events. Useful for "
                "finding precedents ('Has this attack type hit this slice before? What worked?'). "
                "All filter parameters are optional."
            ),
            "parameters": _model_to_openai_schema(QueryHistoryInput),
        },
    },
]


def get_anthropic_schemas() -> List[Dict[str, Any]]:
    """Return tool schemas in Anthropic ``tool_use`` format."""
    result = []
    for tool in TOOL_SCHEMAS_OPENAI:
        fn = tool["function"]
        result.append(
            {
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            }
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Tool dispatcher
# ──────────────────────────────────────────────────────────────────────────────

_ATTACK_NAME_TO_NUMBER = {
    "benign": 0,
    "syn_flood": 1,
    "udp_flood": 2,
    "icmp_flood": 3,
    "dns_flood": 4,
    "gtp_u_flood": 5,
}


async def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Route a tool call (name + args dict) to its async implementation.

    Bridges the simplified agent-facing parameter set to the actual tool
    function signatures in ``src/tools/``.
    """
    # Lazy imports to avoid circular deps and allow demo mode
    from src.tools.detect_anomaly import detect_anomaly
    from src.tools.explain_attack import explain_attack
    from src.tools.recommend_response import recommend_response
    from src.tools.query_history import query_history

    logger.debug("execute_tool: %s(%s)", name, json.dumps(args, default=str)[:200])

    if name == "detect_anomaly":
        return await detect_anomaly(**{k: v for k, v in args.items() if v is not None})

    elif name == "explain_attack":
        # Build a minimal detection_result dict from the flat agent args
        attack_type = args.get("attack_type", "unknown")
        severity    = args.get("severity", "medium")
        detection_result = {
            "attack_type":        attack_type,
            "severity":           severity,
            "is_attack":          attack_type != "benign",
            "binary_probability": args.get("binary_probability", 0.8),
            "attack_number":      _ATTACK_NAME_TO_NUMBER.get(attack_type, -1),
            "summary":            f"{attack_type.replace('_', ' ').title()} detected",
            "features_used":      [],
        }
        cells = args.get("affected_cells")
        cell_list = cells.split(",") if isinstance(cells, str) else cells
        return await explain_attack(
            detection_result=detection_result,
            ue_identity=args.get("ue_id"),
            imeisv=args.get("imeisv"),
            ue_ip=args.get("ue_ip"),
            affected_cells=cell_list,
            slice_type=args.get("slice_type"),
        )

    elif name == "recommend_response":
        cells = args.get("affected_cells")
        cell_list = cells.split(",") if isinstance(cells, str) else cells
        return await recommend_response(
            attack_type=args.get("attack_type", "unknown"),
            severity=args.get("severity", "medium"),
            slice_type=args.get("slice_type"),
            imeisv=args.get("imeisv"),
            ue_ip=args.get("ue_ip"),
            affected_cells=cell_list,
            allow_auto_execute=args.get("auto_execute", False),
        )

    elif name == "query_history":
        return await query_history(
            attack_type=args.get("attack_type"),
            severity=args.get("severity"),
            slice_type=args.get("slice_type"),
            cell_id=args.get("cell_id"),
            ue_ip=args.get("ue_ip"),
            imeisv=args.get("imeisv"),
            time_range_hours=args.get("time_range_hours", 24),
            limit=args.get("limit", 10),
        )

    else:
        raise ValueError(f"Unknown tool: {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# LangChain StructuredTool adapter
# ──────────────────────────────────────────────────────────────────────────────

def get_langchain_tools():  # type: ignore[return]
    """Return a list of LangChain ``StructuredTool`` objects for all four tools.

    Each tool's ``args_schema`` is the corresponding Pydantic model defined
    above so that LangGraph's ReAct agent gets strongly-typed input schemas.

    Raises ``ImportError`` if ``langchain`` / ``langchain-core`` are not installed.
    """
    try:
        from langchain.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "LangChain is required for this agent backend: "
            "pip install langchain langchain-core langgraph"
        ) from exc

    def _make_tool(tool_name: str, model: type[BaseModel], description: str) -> StructuredTool:
        def _sync_run(**kwargs: Any) -> str:
            result = asyncio.get_event_loop().run_until_complete(
                execute_tool(tool_name, kwargs)
            )
            return json.dumps(result, default=str, indent=2)

        async def _async_run(**kwargs: Any) -> str:
            result = await execute_tool(tool_name, kwargs)
            return json.dumps(result, default=str, indent=2)

        return StructuredTool.from_function(
            func=_sync_run,
            coroutine=_async_run,
            name=tool_name,
            description=description,
            args_schema=model,
        )

    schemas_by_name = {t["function"]["name"]: t["function"] for t in TOOL_SCHEMAS_OPENAI}

    return [
        _make_tool("detect_anomaly",     DetectAnomalyInput,    schemas_by_name["detect_anomaly"]["description"]),
        _make_tool("explain_attack",     ExplainAttackInput,    schemas_by_name["explain_attack"]["description"]),
        _make_tool("recommend_response", RecommendResponseInput, schemas_by_name["recommend_response"]["description"]),
        _make_tool("query_history",      QueryHistoryInput,     schemas_by_name["query_history"]["description"]),
    ]
