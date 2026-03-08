"""
MCP Tool: explain_attack

Prompts the LLM to generate a natural-language incident report
based on a detection result, UE identity, and optional slice context.

The LLM is grounded in dataset knowledge via a detailed system prompt
so it can produce accurate, context-aware explanations without hallucinating
5G-specific protocol details.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a 5G network security analyst specialising in DDoS detection on 3GPP-compliant networks.
You have deep knowledge of the NCSRD-DS-5GDDoS dataset (v3.0), which captures the following attack types
on a physical 5G testbed with 9 UEs (5 malicious, 4 benign):

  Attack Types and 5G Impact:
  - SYN Flooding   : TCP handshake exhaustion → targets AMF/SMF signaling plane (HIGH severity)
  - UDP Flooding   : Stateless packet flooding → saturates UPF data plane (HIGH severity)
  - ICMP Flooding  : Echo-request overload → affects all network layers (MEDIUM severity)
  - DNS Flooding   : DNS resolver amplification → targets edge DNS (MEDIUM-HIGH severity)
  - GTP-U Flooding : 5G tunnel flooding → CRITICAL, unique to 5G core architecture

  Network Slices Affected:
  - eMBB: UDP/GTP-U flooding targets high-bandwidth users
  - URLLC: SYN flooding disrupts low-latency sessions
  - mMTC: ICMP/DNS flooding affects IoT device density

  Testbed: Amarisoft Classic/Mini, Open5GS core, 3 cells, 9 UEs, InfluxDB telemetry.

When generating incident reports:
  1. Be precise about which 5G layer is affected (RAN / Core / NAS)
  2. Cite specific metrics that triggered the detection (e.g., ul_retx spike, gnb_cpu_load)
  3. Recommend targeted mitigations matching the attack type and slice
  4. Use EU project terminology where appropriate (PRIVATEER, OASEES)
  5. Format output as a structured incident report
"""


async def explain_attack(
    detection_result: Dict[str, Any],
    ue_identity: Optional[str] = None,
    imeisv: Optional[str] = None,
    ue_ip: Optional[str] = None,
    affected_cells: Optional[str] = None,
    slice_type: Optional[str] = None,
    timestamp: Optional[str] = None,
    include_remediation: bool = True,
) -> Dict[str, Any]:
    """
    Generate a natural-language incident report for a detected DDoS event.

    Parameters
    ----------
    detection_result    : Output dict from detect_anomaly tool
    ue_identity         : UE identifier string (5G-TMSI, RNTI, or internal ID)
    imeisv              : International Mobile Equipment Identity Software Version
    ue_ip               : Source IP address of the attacking UE
    affected_cells      : Comma-separated cell IDs affected (e.g. "1,3")
    slice_type          : Network slice type ("eMBB", "URLLC", "mMTC")
    timestamp           : ISO 8601 event timestamp
    include_remediation : Include recommended mitigation actions in the report

    Returns
    -------
    dict with keys:
        report_text     : Full incident report as markdown string
        attack_type     : Confirmed attack type
        severity        : Severity string
        affected_layer  : 5G protocol layer (RAN / Core / NAS / All)
        key_metrics     : Dict of the metrics that triggered the detection
    """
    attack_type = detection_result.get("attack_type", "unknown")
    severity    = detection_result.get("severity", "unknown")
    probability = detection_result.get("binary_probability", 0.0)
    is_attack   = detection_result.get("is_attack", False)

    if not is_attack:
        return {
            "report_text":    "No attack detected – no incident report generated.",
            "attack_type":    "benign",
            "severity":       "none",
            "affected_layer": "none",
            "key_metrics":    {},
        }

    # Build the prompt
    context_lines = [
        f"## Incident Detection Summary",
        f"- **Attack Type:**       {attack_type.replace('_', ' ').title()}",
        f"- **Severity:**          {severity.upper()}",
        f"- **Confidence:**        {probability:.1%}",
        f"- **Timestamp:**         {timestamp or 'unknown'}",
        f"- **UE Identity:**       {ue_identity or 'unknown'}",
        f"- **IMEISV:**            {imeisv or 'unknown'}",
        f"- **Source IP:**         {ue_ip or 'unknown'}",
        f"- **Affected Cells:**    {affected_cells or 'unknown'}",
        f"- **Slice Type:**        {slice_type or 'unknown'}",
        "",
        "## Key Telemetry Metrics at Time of Detection",
    ]

    key_metrics = {}
    features = detection_result.get("features_used", [])
    # Pull notable metrics from the detection result if they were passed as context
    if "ul_bitrate" in str(detection_result):
        context_lines.append("(Metrics embedded in detection result – see raw values below)")

    context_lines += [
        "",
        f"## Class Probability Breakdown",
        str(detection_result.get("class_probabilities", {})),
        "",
    ]

    if include_remediation:
        context_lines.append(
            "Please generate a full incident report including: "
            "(1) executive summary, (2) technical analysis of the attack vector, "
            "(3) affected 5G protocol layers, (4) timeline, "
            "(5) specific mitigation recommendations for this slice type."
        )
    else:
        context_lines.append(
            "Please generate a concise incident report covering: "
            "executive summary, technical attack vector analysis, and affected 5G layers."
        )

    prompt = "\n".join(context_lines)

    try:
        from src.models.llm_client import get_llm_client
        from src.config import llm_config

        client = get_llm_client()
        report_text = await client.complete(
            prompt     = prompt,
            system     = SYSTEM_PROMPT,
            max_tokens = llm_config.max_tokens,
            temperature = 0.2,  # Low temperature for factual reports
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        report_text = _fallback_report(attack_type, severity, probability, ue_identity, timestamp)

    # Determine affected 5G layer
    layer_map = {
        "syn_flood":   "Core (AMF/SMF signaling plane)",
        "udp_flood":   "Core (UPF data plane)",
        "icmp_flood":  "All layers (RAN + Core + NAS)",
        "dns_flood":   "Edge DNS / Core",
        "gtp_u_flood": "Core (GTP-U tunnel – 5G-specific)",
        "benign":      "None",
    }
    affected_layer = layer_map.get(attack_type, "Unknown")

    return {
        "report_text":    report_text,
        "attack_type":    attack_type,
        "severity":       severity,
        "affected_layer": affected_layer,
        "key_metrics":    key_metrics,
        "ue_identity":    ue_identity,
        "imeisv":         imeisv,
        "ue_ip":          ue_ip,
        "timestamp":      timestamp,
    }


def _fallback_report(
    attack_type: str,
    severity: str,
    probability: float,
    ue_id: Optional[str],
    timestamp: Optional[str],
) -> str:
    return f"""
# 5G DDoS Incident Report

**Incident ID:** AUTO-{timestamp or 'UNKNOWN'}
**UE:**         {ue_id or 'Unknown'}
**Attack Type:** {attack_type.replace('_', ' ').title()}
**Severity:**    {severity.upper()}
**Confidence:**  {probability:.1%}

## Summary
A {attack_type.replace('_', ' ')} attack was detected with {probability:.1%} confidence.
This attack type has a severity rating of **{severity}**.

## Technical Details
See `detect_anomaly` output for full feature breakdown.

## Recommended Actions
1. Isolate the offending UE from the affected slice.
2. Rate-limit or blacklist the source IP.
3. Monitor adjacent cells for lateral spread.
4. Escalate to NOC if severity is CRITICAL or HIGH.

*Note: Full LLM-powered report unavailable – LLM backend not configured.*
""".strip()
