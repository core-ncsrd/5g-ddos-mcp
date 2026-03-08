"""
MCP Tool: query_history

Searches the historical incident database (seeded from the NCSRD-DS-5GDDoS dataset)
for similar attack patterns, enabling few-shot in-context reasoning.

The incident database (models/incident_history.json) is built by running:
    python scripts/build_incident_db.py

Until the model is trained and the database is seeded, this tool uses
dataset knowledge embedded in the system prompt.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import ATTACK_LABELS, ATTACK_SEVERITY, INCIDENT_DB_PATH

logger = logging.getLogger(__name__)


def _load_incident_db() -> List[Dict[str, Any]]:
    """Load incidents from JSON file. Returns empty list if not found."""
    if not INCIDENT_DB_PATH.exists():
        logger.info("Incident DB not found at %s – returning dataset reference records", INCIDENT_DB_PATH)
        return _get_reference_incidents()
    with open(INCIDENT_DB_PATH, "r") as f:
        return json.load(f)


def _save_incident(incident: Dict[str, Any]) -> None:
    """Append a new incident to the JSON database."""
    INCIDENT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    incidents = _load_incident_db()
    # Remove reference records before saving
    incidents = [i for i in incidents if not i.get("is_reference")]
    incidents.append(incident)
    with open(INCIDENT_DB_PATH, "w") as f:
        json.dump(incidents, f, indent=2, default=str)


async def query_history(
    attack_type:    Optional[str] = None,
    severity:       Optional[str] = None,
    slice_type:     Optional[str] = None,
    cell_id:        Optional[int] = None,
    ue_ip:          Optional[str] = None,
    imeisv:         Optional[str] = None,
    limit:          int = 10,
    include_benign: bool = False,
    time_range_hours: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Search historical incident database for similar attack patterns.

    Used to enable few-shot in-context LLM reasoning:
    "Have we seen this attack type on this slice before? What worked?"

    Parameters
    ----------
    attack_type       : Filter by attack type (e.g. "gtp_u_flood")
    severity          : Filter by severity level
    slice_type        : Filter by slice type ("eMBB", "URLLC", "mMTC")
    cell_id           : Filter by cell ID (1, 2, 3)
    ue_ip             : Filter by source IP address
    imeisv            : Filter by IMEISV (device identity)
    limit             : Maximum number of results to return
    include_benign    : Include benign baseline records
    time_range_hours  : Only return incidents from last N hours

    Returns
    -------
    dict with keys:
        incidents       : List of matching historical incidents
        total_found     : Total count before limit
        summary         : Text summary of patterns found
        attack_frequency: Dict of attack_type → count in history
        dataset_context : Reference data from NCSRD-DS-5GDDoS paper
    """
    all_incidents = _load_incident_db()

    # ── Filter ────────────────────────────────────────────────────────────
    filtered = all_incidents

    if not include_benign:
        filtered = [i for i in filtered if i.get("attack_type", "benign") != "benign"]

    if attack_type:
        filtered = [i for i in filtered if i.get("attack_type") == attack_type]

    if severity:
        filtered = [i for i in filtered if i.get("severity") == severity]

    if slice_type:
        filtered = [i for i in filtered if i.get("slice_type") == slice_type]

    if cell_id is not None:
        filtered = [i for i in filtered
                    if cell_id in (i.get("affected_cells") or [])]

    if ue_ip:
        filtered = [i for i in filtered if i.get("ue_ip") == ue_ip]

    if imeisv:
        filtered = [i for i in filtered if i.get("imeisv") == imeisv]

    if time_range_hours:
        cutoff = datetime.utcnow().timestamp() - time_range_hours * 3600
        filtered = [
            i for i in filtered
            if i.get("timestamp_unix", 0) >= cutoff
        ]

    total_found = len(filtered)

    # ── Sort by timestamp descending (most recent first) ──────────────────
    filtered.sort(key=lambda i: i.get("timestamp_unix", 0), reverse=True)
    incidents = filtered[:limit]

    # ── Frequency analysis ────────────────────────────────────────────────
    attack_freq: Dict[str, int] = {}
    for inc in all_incidents:
        at = inc.get("attack_type", "unknown")
        attack_freq[at] = attack_freq.get(at, 0) + 1

    # ── Summary ───────────────────────────────────────────────────────────
    summary = _build_summary(incidents, total_found, attack_type)

    return {
        "incidents":        incidents,
        "total_found":      total_found,
        "summary":          summary,
        "attack_frequency": attack_freq,
        "dataset_context":  _get_dataset_context(attack_type),
    }


async def record_incident(incident: Dict[str, Any]) -> Dict[str, str]:
    """
    Record a new incident in the history database.
    Called automatically after a confirmed detection.
    """
    # Add metadata
    incident["timestamp_unix"] = datetime.utcnow().timestamp()
    incident["recorded_at"]    = datetime.utcnow().isoformat()

    _save_incident(incident)
    logger.info("Recorded incident: %s at %s", incident.get("attack_type"), incident.get("recorded_at"))
    return {"status": "recorded", "id": incident.get("recorded_at", "unknown")}


def _build_summary(
    incidents: List[Dict],
    total: int,
    attack_type: Optional[str],
) -> str:
    if not incidents:
        return (
            f"No historical incidents found"
            + (f" for attack type '{attack_type}'" if attack_type else "")
            + ". This may be a novel attack pattern."
        )

    types     = set(i.get("attack_type") for i in incidents)
    severities = set(i.get("severity") for i in incidents)
    slices    = set(i.get("slice_type") for i in incidents if i.get("slice_type"))

    return (
        f"Found {total} historical incident(s)"
        + (f" of type '{attack_type}'" if attack_type else "")
        + f". Attack types: {', '.join(filter(None, types))}. "
        f"Severities: {', '.join(filter(None, severities))}. "
        + (f"Affected slices: {', '.join(slices)}." if slices else "")
    )


def _get_dataset_context(attack_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Return relevant reference data from the NCSRD-DS-5GDDoS dataset knowledge.
    This grounds the LLM's few-shot reasoning in empirical data.
    """
    context = {
        "dataset":   "NCSRD-DS-5GDDoS v3.0",
        "doi":       "10.5281/zenodo.13900057",
        "testbed":   "Physical 3GPP-compliant 5G (Amarisoft), 3 cells, 9 UEs",
        "published": "October 7, 2024",
        "license":   "CC BY 4.0",
        "published_results": {
            "best_accuracy": "98% (XGBoost + Gradient Boosting federated, ScienceDirect 2025)",
            "auc_roc":       "99.9%",
            "balancing":     "SMOTE-Tomek proven effective for class imbalance",
        },
    }

    if attack_type:
        attack_profiles = {
            "syn_flood": {
                "mechanism":    "TCP handshake exhaustion",
                "target_layer": "AMF/SMF signaling plane",
                "key_indicators": ["high ul_retx", "pdu_session_setup spikes", "abnormal_release_count"],
                "observed_devices": ["Raspberry Pi 4 + 5G Hat", "Industrial 5G Router"],
            },
            "udp_flood": {
                "mechanism":    "Stateless packet flooding",
                "target_layer": "UPF data plane",
                "key_indicators": ["extreme ul_bitrate", "high gnb_cpu_load"],
                "observed_devices": ["Huawei P40", "D-Link DWR-2101"],
            },
            "icmp_flood": {
                "mechanism":    "Echo-request overload",
                "target_layer": "All layers",
                "key_indicators": ["moderate ul_bitrate spike", "increased paging_attempts"],
            },
            "dns_flood": {
                "mechanism":    "DNS resolver amplification",
                "target_layer": "Edge DNS / Core",
                "key_indicators": ["pdu_session_setup_request bursts", "DNS query volume"],
            },
            "gtp_u_flood": {
                "mechanism":    "5G tunnel flooding",
                "target_layer": "GTP-U tunnel in 5G core",
                "key_indicators": ["gnb_cpu_load critical", "enb_dl/ul_bitrate saturation", "all UE metrics affected"],
                "note":         "CRITICAL – unique to 5G architecture, no equivalent in 4G LTE",
            },
        }
        context["attack_profile"] = attack_profiles.get(attack_type, {})

    return context


def _get_reference_incidents() -> List[Dict[str, Any]]:
    """
    Synthetic reference incidents derived from dataset knowledge.
    Used when the incident DB has not been seeded yet.
    """
    return [
        {
            "id":            "ref-001",
            "attack_type":   "gtp_u_flood",
            "severity":      "critical",
            "slice_type":    "eMBB",
            "cell_id":       1,
            "imeisv":        "86620003000000",
            "ue_ip":         "192.168.1.101",
            "confidence":    0.97,
            "mitigation":    "Slice isolation + UE blacklist",
            "timestamp_unix": 0,
            "is_reference":  True,
            "source":        "NCSRD-DS-5GDDoS dataset reference",
        },
        {
            "id":            "ref-002",
            "attack_type":   "syn_flood",
            "severity":      "high",
            "slice_type":    "URLLC",
            "cell_id":       3,
            "imeisv":        "35737213000000",
            "ue_ip":         "192.168.1.102",
            "confidence":    0.94,
            "mitigation":    "Priority session protection + fast-path blacklist",
            "timestamp_unix": 0,
            "is_reference":  True,
            "source":        "NCSRD-DS-5GDDoS dataset reference",
        },
        {
            "id":            "ref-003",
            "attack_type":   "udp_flood",
            "severity":      "high",
            "slice_type":    "eMBB",
            "cell_id":       2,
            "confidence":    0.91,
            "mitigation":    "Rate limiting + UE quarantine",
            "timestamp_unix": 0,
            "is_reference":  True,
            "source":        "NCSRD-DS-5GDDoS dataset reference",
        },
        {
            "id":            "ref-004",
            "attack_type":   "icmp_flood",
            "severity":      "medium",
            "slice_type":    "mMTC",
            "confidence":    0.88,
            "mitigation":    "Per-device anomaly scoring + collective rate limit",
            "timestamp_unix": 0,
            "is_reference":  True,
            "source":        "NCSRD-DS-5GDDoS dataset reference",
        },
        {
            "id":            "ref-005",
            "attack_type":   "dns_flood",
            "severity":      "medium-high",
            "slice_type":    "eMBB",
            "confidence":    0.90,
            "mitigation":    "DNS resolver isolation + IP rate limit",
            "timestamp_unix": 0,
            "is_reference":  True,
            "source":        "NCSRD-DS-5GDDoS dataset reference",
        },
    ]
