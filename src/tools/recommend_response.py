"""
MCP Tool: recommend_response

The LLM reasons over current slice topology, attack severity, and SLA requirements
to recommend targeted mitigation actions:
  - Slice isolation
  - UE blacklisting
  - Rate limiting
  - Traffic rerouting

Optionally auto-executes actions via a generic REST API if RESPONSE_API_URL is set.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """\
You are an automated 5G security response orchestrator for a 3GPP-compliant network.
Your role is to recommend precise, proportional mitigation actions for DDoS attacks
based on the attack type, slice configuration, and current network state.

You must:
1. Recommend slice-aware actions (eMBB / URLLC / mMTC have different SLAs)
2. Prioritise minimal disruption to benign UEs
3. Provide generic REST API call examples for each action
4. Rate actions by effectiveness and disruption level

Available response primitives:
  - isolate_slice(slice_id)           : Full slice isolation
  - blacklist_ue(imeisv, cell_id)     : Block specific UE
  - rate_limit_ue(imeisv, max_bps)    : Throttle UE bandwidth
  - reroute_traffic(slice_id, path)   : Divert slice traffic
  - alert_noc(severity, message)      : Notify NOC operators

Slice SLA requirements:
  - eMBB   : High bandwidth, latency < 20ms, disruption acceptable
  - URLLC  : Ultra-reliable, latency < 1ms, MINIMAL disruption
  - mMTC   : High device density, aggregate rate limiting preferred
"""


async def recommend_response(
    attack_type:        str,
    severity:           str,
    slice_type:         Optional[str] = None,
    affected_ue_ids:    Optional[List[str]] = None,
    affected_cells:     Optional[List[int]] = None,
    imeisv:             Optional[str] = None,
    ue_ip:              Optional[str] = None,
    current_ue_count:   int = 0,
    active_slice_ids:   Optional[List[str]] = None,
    sla_requirements:   Optional[Dict[str, Any]] = None,
    allow_auto_execute: bool = False,
) -> Dict[str, Any]:
    """
    Recommend targeted mitigation actions for a detected 5G DDoS attack.

    Parameters
    ----------
    attack_type         : e.g. "gtp_u_flood", "syn_flood", "udp_flood"
    severity            : "none" / "medium" / "medium-high" / "high" / "critical"
    slice_type          : "eMBB" / "URLLC" / "mMTC" (affects SLA constraints)
    affected_ue_ids     : List of UE identifiers involved
    affected_cells      : List of affected cell IDs (1, 2, 3)
    imeisv              : IMEISV of primary attacking UE
    ue_ip               : IP address of attacking UE
    current_ue_count    : Total UEs on the slice (for proportionality)
    active_slice_ids    : Currently active slice IDs
    sla_requirements    : Custom SLA dict (overrides defaults)
    allow_auto_execute  : If True and RESPONSE_API_URL is set, execute actions

    Returns
    -------
    dict with keys:
        recommendations : List of recommended actions with priority
        rationale       : LLM-generated explanation
        auto_executed   : List of actions that were auto-executed (if any)
        api_commands    : Example REST API call payloads for each action
    """
    prompt_lines = [
        "## Active Incident",
        f"- Attack Type:    {attack_type.replace('_', ' ').title()}",
        f"- Severity:       {severity.upper()}",
        f"- Slice Type:     {slice_type or 'unknown'}",
        f"- Affected Cells: {affected_cells or 'unknown'}",
        f"- Attacking UE:   {imeisv or affected_ue_ids or 'unknown'}",
        f"- Source IP:      {ue_ip or 'unknown'}",
        f"- UEs on Slice:   {current_ue_count}",
        f"- Active Slices:  {active_slice_ids or 'not specified'}",
        "",
        "## Task",
        "Recommend specific mitigation actions for this attack. For each action, provide:",
        "  1. Action name and parameters",
        "  2. Priority (immediate / short-term / long-term)",
        "  3. Effectiveness estimate",
        "  4. Disruption level to benign UEs",
        "",
        "Also provide generic REST API call examples for each action.",
    ]

    prompt = "\n".join(prompt_lines)

    auto_executed = []
    api_commands = _build_api_commands(attack_type, slice_type, imeisv, ue_ip, affected_cells)

    try:
        from src.models.llm_client import get_llm_client
        from src.config import llm_config

        client = get_llm_client()
        rationale = await client.complete(
            prompt      = prompt,
            system      = RESPONSE_SYSTEM_PROMPT,
            max_tokens  = llm_config.max_tokens,
            temperature = 0.2,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        rationale = _rule_based_recommendations(attack_type, severity, slice_type)

    recommendations = _extract_recommendations(attack_type, severity, slice_type)

    if allow_auto_execute:
        auto_executed = await _auto_execute(recommendations, imeisv, ue_ip)

    return {
        "recommendations": recommendations,
        "rationale":       rationale,
        "auto_executed":   auto_executed,
        "api_commands":    api_commands,
        "attack_type":     attack_type,
        "severity":        severity,
        "slice_type":      slice_type,
    }


def _extract_recommendations(
    attack_type: str,
    severity:    str,
    slice_type:  Optional[str],
) -> List[Dict[str, Any]]:
    """Rule-based baseline recommendations (LLM enriches these)."""
    recs = []

    if severity in ("high", "critical"):
        recs.append({
            "action":        "blacklist_ue",
            "priority":      "immediate",
            "description":   "Block the attacking UE at the gNB level",
            "disruption":    "low (targeted)",
            "effectiveness": "high",
        })

    recs.append({
        "action":        "rate_limit_ue",
        "priority":      "immediate",
        "description":   "Apply bandwidth cap to the suspicious UE",
        "disruption":    "low",
        "effectiveness": "medium-high",
    })

    if attack_type in ("gtp_u_flood", "syn_flood") and slice_type != "URLLC":
        recs.append({
            "action":        "isolate_slice",
            "priority":      "short-term",
            "description":   "Isolate affected slice to prevent lateral spread",
            "disruption":    "medium",
            "effectiveness": "very high",
        })

    recs.append({
        "action":        "alert_noc",
        "priority":      "immediate",
        "description":   "Notify NOC operators with full incident report",
        "disruption":    "none",
        "effectiveness": "enables manual oversight",
    })

    return recs


def _build_api_commands(
    attack_type:    str,
    slice_type:     Optional[str],
    imeisv:         Optional[str],
    ue_ip:          Optional[str],
    affected_cells: Optional[List[int]],
) -> List[Dict[str, str]]:
    """
    Build generic REST API call examples for each mitigation action.
    These are illustrative payloads — adapt endpoints to your network
    management system (Open5GS, free5GC, OAI, or any custom NMS).
    """
    commands = []

    if ue_ip:
        commands.append({
            "action":      "rate_limit_ue",
            "description": "Throttle UE uplink/downlink bandwidth",
            "method":      "POST",
            "endpoint":    "/api/v1/ue/rate-limit",
            "payload":     f'{{"ue_ip": "{ue_ip}", "max_ul_bps": 1000000, "max_dl_bps": 1000000}}',
        })

    if imeisv:
        commands.append({
            "action":      "blacklist_ue",
            "description": "Block UE by device identity",
            "method":      "POST",
            "endpoint":    "/api/v1/ue/blacklist",
            "payload":     f'{{"imeisv": "{imeisv}", "reason": "DDoS attack detected"}}',
        })

    if attack_type in ("gtp_u_flood", "udp_flood", "syn_flood") and slice_type:
        commands.append({
            "action":      "isolate_slice",
            "description": f"Isolate {slice_type} slice from rest of network",
            "method":      "PUT",
            "endpoint":    "/api/v1/slice/{slice_id}/isolate",
            "payload":     f'{{"reason": "{attack_type}", "duration_seconds": 300}}',
        })

    commands.append({
        "action":      "alert_noc",
        "description": "Send alert to NOC webhook",
        "method":      "POST",
        "endpoint":    "/api/v1/alerts",
        "payload":     (
            f'{{"severity": "high", "attack_type": "{attack_type}", '
            f'"ue_ip": "{ue_ip or ""}", "message": "DDoS detected"}}'
        ),
    })

    return commands


def _rule_based_recommendations(
    attack_type: str,
    severity:    str,
    slice_type:  Optional[str],
) -> str:
    return (
        f"Rule-based recommendation for {attack_type.replace('_', ' ')} "
        f"({severity} severity, {slice_type or 'unknown'} slice):\n\n"
        "1. Immediately blacklist the attacking UE.\n"
        "2. Apply rate limiting to the affected cell.\n"
        "3. Monitor for lateral spread to adjacent cells.\n"
        "4. Notify NOC with full incident details.\n\n"
        "(LLM backend unavailable – using rule-based fallback)"
    )


async def _auto_execute(
    recommendations: List[Dict[str, Any]],
    imeisv:          Optional[str],
    ue_ip:           Optional[str],
) -> List[str]:
    """
    Auto-execute immediate-priority actions via a generic REST API.
    Configure with RESPONSE_API_URL and RESPONSE_API_KEY environment variables.
    """
    import os
    import aiohttp

    api_url = os.getenv("RESPONSE_API_URL", "")
    api_key = os.getenv("RESPONSE_API_KEY", "")

    if not api_url:
        logger.warning("RESPONSE_API_URL not configured – skipping auto-execution")
        return []

    executed = []
    immediate = [r for r in recommendations if r.get("priority") == "immediate"]
    headers   = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession(headers=headers) as session:
        for rec in immediate:
            try:
                url = f"{api_url}{rec.get('endpoint', '')}"
                async with session.post(url, json=rec, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status < 300:
                        executed.append(rec["action"])
                        logger.info("Auto-executed: %s", rec["action"])
                    else:
                        logger.warning("Response API error %d for action %s", resp.status, rec["action"])
            except Exception as exc:
                logger.error("Failed to auto-execute %s: %s", rec["action"], exc)

    return executed
