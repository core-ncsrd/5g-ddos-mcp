"""
MCP Tool: detect_anomaly

Accepts a real-time 5G telemetry snapshot and returns:
  - Attack probability (binary)
  - Attack type classification (multi-class: syn/udp/icmp/dns/gtp-u)
  - Confidence scores
  - Severity assessment

Calls the trained XGBoost classifier (models/xgb_*.pkl).
Falls back to rule-based heuristics if no model is loaded.
"""

import logging
from typing import Any, Dict, Optional

from src.config import ATTACK_LABELS, ATTACK_SEVERITY, server_config

logger = logging.getLogger(__name__)


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
) -> Dict[str, Any]:
    """
    Run DDoS anomaly detection on a 5G telemetry snapshot.

    Inputs map directly to the NCSRD-DS-5GDDoS feature schema:
      - UE-level:  ul/dl_bitrate, ul/dl_retx, MCS, PHR, SNR metrics
      - eNB-level: cell bitrates, usage ratios, gNB CPU load
      - MME-level: NAS counters (PDU sessions, paging, abnormal releases)

    Returns a structured detection result with attack type and severity.
    """
    features = {
        "ul_bitrate":                 ul_bitrate,
        "dl_bitrate":                 dl_bitrate,
        "ul_retx":                    ul_retx,
        "dl_retx":                    dl_retx,
        "ul_mcs":                     ul_mcs,
        "dl_mcs":                     dl_mcs,
        "ul_phr":                     ul_phr,
        "cell_id":                    float(cell_id),
        "pucch_snr":                  pucch_snr,
        "pusch_snr":                  pusch_snr,
        "enb_dl_bitrate":             enb_dl_bitrate,
        "enb_ul_bitrate":             enb_ul_bitrate,
        "enb_dl_usage_ratio":         enb_dl_usage_ratio,
        "enb_ul_usage_ratio":         enb_ul_usage_ratio,
        "gnb_cpu_load":               gnb_cpu_load,
        "pdu_session_setup_request":  pdu_session_setup_request,
        "pdu_session_setup_response": pdu_session_setup_response,
        "paging_attempts":            paging_attempts,
        "context_setup_request":      context_setup_request,
        "context_setup_response":     context_setup_response,
        "abnormal_release_count":     abnormal_release_count,
    }

    try:
        from src.models.classifier import get_classifier
        clf    = get_classifier()
        result = clf.predict(features)
    except Exception as exc:
        logger.warning("ML classifier unavailable (%s) – using heuristics", exc)
        result = _heuristic_detection(features)

    # Attach metadata
    result["ue_id"]     = ue_id
    result["timestamp"] = timestamp
    result["threshold"] = server_config.anomaly_threshold

    # Human-readable summary
    if result["is_attack"]:
        result["summary"] = (
            f"ALERT: {result['attack_type'].upper()} detected on UE {ue_id or 'unknown'} "
            f"(confidence {result['binary_probability']:.1%}, severity: {result['severity']})"
        )
    else:
        result["summary"] = (
            f"NORMAL: Traffic on UE {ue_id or 'unknown'} is benign "
            f"(attack probability {result['binary_probability']:.1%})"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based heuristics  (fallback when model is not trained yet)
# ─────────────────────────────────────────────────────────────────────────────
_THRESHOLDS = {
    "ul_bitrate":            50_000_000,   # 50 Mbps sustained UL → suspicious
    "ul_retx":               500,           # High retransmission → SYN / UDP flood
    "gnb_cpu_load":          90.0,          # gNB CPU spike → GTP-U / flooding
    "abnormal_release_count": 10,           # MME NAS anomaly
    "pdu_session_setup_request": 100,       # Mass PDU setup → DNS / SYN flood
}

def _heuristic_detection(features: Dict[str, float]) -> Dict[str, Any]:
    score = 0.0
    attack_type = "benign"

    if features.get("ul_bitrate", 0) > _THRESHOLDS["ul_bitrate"]:
        score += 0.4
        attack_type = "udp_flood"
    if features.get("ul_retx", 0) > _THRESHOLDS["ul_retx"]:
        score += 0.3
        attack_type = "syn_flood"
    if features.get("gnb_cpu_load", 0) > _THRESHOLDS["gnb_cpu_load"]:
        score += 0.5
        attack_type = "gtp_u_flood"
    if features.get("abnormal_release_count", 0) > _THRESHOLDS["abnormal_release_count"]:
        score += 0.3
    if features.get("pdu_session_setup_request", 0) > _THRESHOLDS["pdu_session_setup_request"]:
        score += 0.35
        attack_type = "dns_flood"

    score = min(score, 1.0)
    is_attack = score >= 0.5

    attack_number = 0
    if is_attack:
        from src.config import ATTACK_LABELS_INV
        attack_number = ATTACK_LABELS_INV.get(attack_type, 0)

    return {
        "is_attack":           is_attack,
        "attack_type":         attack_type if is_attack else "benign",
        "attack_number":       attack_number if is_attack else 0,
        "binary_probability":  round(score, 4),
        "class_probabilities": {lbl: 0.0 for lbl in ATTACK_LABELS.values()},
        "severity":            ATTACK_SEVERITY.get(attack_type if is_attack else "benign", "none"),
        "features_used":       list(features.keys()),
        "mode":                "heuristic",
    }
