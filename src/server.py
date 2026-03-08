"""
5G-DDoS MCP Server
==================
A Model Context Protocol server for real-time 5G DDoS detection and response,
backed by the NCSRD-DS-5GDDoS dataset and supporting:

  LLM Backends  : Claude (Anthropic) | Ollama (local OSS) | OpenAI-compatible
  ML Model      : XGBoost classifier trained on NCSRD-DS-5GDDoS v3.0
  MCP Tools     : detect_anomaly | explain_attack | recommend_response |
                  query_history

Usage:
    # With uv (recommended)
    uv run python -m src.server

    # With pip
    python -m src.server

    # Inspect with MCP Inspector
    npx @modelcontextprotocol/inspector uv run python -m src.server

Environment variables → see .env.example
"""

import asyncio
import json
import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from src.config import ATTACK_LABELS, server_config

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = getattr(logging, server_config.log_level, logging.INFO),
    format  = "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt = "%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FastMCP instance
# ─────────────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name        = "5g-ddos-mcp",
    description = (
        "5G DDoS detection and response server powered by the NCSRD-DS-5GDDoS dataset. "
        "Supports Claude, Ollama (llama3, mistral, phi4, …), and OpenAI-compatible LLMs."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: detect_anomaly
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Detect 5G DDoS attacks from real-time UE telemetry. "
        "Accepts per-UE radio metrics (bitrate, retransmissions, MCS, SNR), "
        "eNB cell counters, and MME NAS counters. "
        "Returns attack probability, type (syn/udp/icmp/dns/gtp_u), and severity. "
        "Feature schema matches NCSRD-DS-5GDDoS dataset (DOI: 10.5281/zenodo.13900057)."
    )
)
async def detect_anomaly(
    ul_bitrate:                  float = 0.0,
    dl_bitrate:                  float = 0.0,
    ul_retx:                     float = 0.0,
    dl_retx:                     float = 0.0,
    ul_mcs:                      float = 0.0,
    dl_mcs:                      float = 0.0,
    ul_phr:                      float = 0.0,
    cell_id:                     int   = 0,
    pucch_snr:                   float = 0.0,
    pusch_snr:                   float = 0.0,
    enb_dl_bitrate:              float = 0.0,
    enb_ul_bitrate:              float = 0.0,
    enb_dl_usage_ratio:          float = 0.0,
    enb_ul_usage_ratio:          float = 0.0,
    gnb_cpu_load:                float = 0.0,
    pdu_session_setup_request:   float = 0.0,
    pdu_session_setup_response:  float = 0.0,
    paging_attempts:             float = 0.0,
    context_setup_request:       float = 0.0,
    context_setup_response:      float = 0.0,
    abnormal_release_count:      float = 0.0,
    ue_id:                       Optional[str] = None,
    timestamp:                   Optional[str] = None,
) -> str:
    from src.tools.detect_anomaly import detect_anomaly as _detect
    result = await _detect(**{k: v for k, v in locals().items()})
    return json.dumps(result, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: explain_attack
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Generate a natural-language incident report for a detected 5G DDoS event. "
        "Uses the configured LLM (Claude / Ollama / OpenAI-compatible) to produce "
        "a structured report covering: attack vector, affected 5G layers, timeline, "
        "and remediation recommendations. "
        "Pass the output of detect_anomaly as detection_result."
    )
)
async def explain_attack(
    detection_result:    str,
    ue_identity:         Optional[str] = None,
    imeisv:              Optional[str] = None,
    ue_ip:               Optional[str] = None,
    affected_cells:      Optional[str] = None,
    slice_type:          Optional[str] = None,
    timestamp:           Optional[str] = None,
    include_remediation: bool          = True,
) -> str:
    from src.tools.explain_attack import explain_attack as _explain

    # Parse detection_result if it arrives as a JSON string
    if isinstance(detection_result, str):
        try:
            detection_result = json.loads(detection_result)
        except json.JSONDecodeError:
            return json.dumps({"error": "detection_result must be a valid JSON object"})

    result = await _explain(
        detection_result    = detection_result,
        ue_identity         = ue_identity,
        imeisv              = imeisv,
        ue_ip               = ue_ip,
        affected_cells      = affected_cells,
        slice_type          = slice_type,
        timestamp           = timestamp,
        include_remediation = include_remediation,
    )
    return json.dumps(result, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: recommend_response
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Get LLM-powered mitigation recommendations for a 5G DDoS attack. "
        "The LLM reasons over attack type, slice topology, and SLA requirements "
        "to recommend targeted actions: slice isolation, UE blacklisting, "
        "rate limiting, or traffic rerouting. "
        "Returns generic REST API call examples for each action."
    )
)
async def recommend_response(
    attack_type:        str,
    severity:           str,
    slice_type:         Optional[str] = None,
    affected_ue_ids:    Optional[str] = None,   # JSON array string
    affected_cells:     Optional[str] = None,   # JSON array string
    imeisv:             Optional[str] = None,
    ue_ip:              Optional[str] = None,
    current_ue_count:   int           = 0,
    active_slice_ids:   Optional[str] = None,   # JSON array string
    allow_auto_execute: bool          = False,
) -> str:
    from src.tools.recommend_response import recommend_response as _recommend

    def _parse_list(val: Optional[str]) -> Optional[list]:
        if not val:
            return None
        try:
            return json.loads(val)
        except Exception:
            return [v.strip() for v in val.split(",") if v.strip()]

    result = await _recommend(
        attack_type        = attack_type,
        severity           = severity,
        slice_type         = slice_type,
        affected_ue_ids    = _parse_list(affected_ue_ids),
        affected_cells     = _parse_list(affected_cells),
        imeisv             = imeisv,
        ue_ip              = ue_ip,
        current_ue_count   = current_ue_count,
        active_slice_ids   = _parse_list(active_slice_ids),
        allow_auto_execute = allow_auto_execute,
    )
    return json.dumps(result, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: query_history
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Search historical 5G DDoS incident database for similar attack patterns. "
        "Enables few-shot in-context LLM reasoning: retrieves past incidents "
        "matching the current attack type, slice, cell, or UE identity. "
        "Database is pre-seeded with reference patterns from NCSRD-DS-5GDDoS "
        "and grows with each detected incident."
    )
)
async def query_history(
    attack_type:      Optional[str] = None,
    severity:         Optional[str] = None,
    slice_type:       Optional[str] = None,
    cell_id:          Optional[int] = None,
    ue_ip:            Optional[str] = None,
    imeisv:           Optional[str] = None,
    limit:            int           = 10,
    include_benign:   bool          = False,
    time_range_hours: Optional[int] = None,
) -> str:
    from src.tools.query_history import query_history as _query
    result = await _query(
        attack_type      = attack_type,
        severity         = severity,
        slice_type       = slice_type,
        cell_id          = cell_id,
        ue_ip            = ue_ip,
        imeisv           = imeisv,
        limit            = limit,
        include_benign   = include_benign,
        time_range_hours = time_range_hours,
    )
    return json.dumps(result, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Resource: dataset_info
# ─────────────────────────────────────────────────────────────────────────────
@mcp.resource("dataset://ncsrd-5gddos")
def dataset_info() -> str:
    """Metadata and download instructions for the NCSRD-DS-5GDDoS dataset."""
    from src.data.loader import check_dataset_available
    availability = check_dataset_available()
    info = {
        "name":        "NCSRD-DS-5GDDoS v3.0",
        "doi":         "10.5281/zenodo.13900057",
        "url":         "https://zenodo.org/records/13900057",
        "license":     "CC BY 4.0",
        "size":        "~620 MB (7 files)",
        "published":   "October 7, 2024",
        "description": (
            "High-fidelity 5G DDoS dataset on a physical 3GPP-compliant testbed. "
            "3 cells, 9 UEs (5 malicious, 4 benign). "
            "Attack types: SYN, UDP, ICMP, DNS, GTP-U flooding."
        ),
        "files": availability,
        "download_instructions": "See data/DATASET_INSTRUCTIONS.md",
    }
    return json.dumps(info, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Resource: attack_taxonomy
# ─────────────────────────────────────────────────────────────────────────────
@mcp.resource("taxonomy://5g-ddos-attacks")
def attack_taxonomy() -> str:
    """5G DDoS attack taxonomy from the NCSRD-DS-5GDDoS dataset."""
    from src.config import ATTACK_SEVERITY
    taxonomy = {
        lbl: {
            "attack_number": num,
            "severity":      ATTACK_SEVERITY.get(lbl, "unknown"),
        }
        for num, lbl in ATTACK_LABELS.items()
    }
    return json.dumps(taxonomy, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting 5G-DDoS MCP Server…")
    logger.info("LLM backend:   %s", server_config)
    logger.info("Tools exposed: detect_anomaly | explain_attack | recommend_response | query_history")
    mcp.run()
