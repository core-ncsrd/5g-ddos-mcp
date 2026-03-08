"""
5G-DDoS MCP Server Configuration
Supports: Claude (Anthropic), Ollama (local OSS LLMs), OpenAI-compatible endpoints
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset file names (place downloaded files in ./data/)
DATASET_FILES = {
    "merged":        "amari_ue_data_merged_with_attack_number.csv",   # 241.5 MB – primary ML file
    "ue_classic":    "amari_ue_data_classic_tabular.csv",              # 143.6 MB
    "ue_mini":       "amari_ue_data_mini_tabular.csv",                 # 87.3 MB
    "enb_classic":   "enb_counters_data_classic_tabular.csv",          # 72.2 MB
    "enb_mini":      "enb_counters_data_mini_tabular.csv",             # 38.2 MB
    "mme":           "mme_counters.csv",                               # 37.1 MB
    "summary":       "summary_report.xlsx",                            # 18.0 kB
}

# Trained model artifact paths
MODEL_ARTIFACT_BINARY     = MODELS_DIR / "xgb_binary_classifier.pkl"
MODEL_ARTIFACT_MULTICLASS = MODELS_DIR / "xgb_multiclass_classifier.pkl"
SCALER_ARTIFACT           = MODELS_DIR / "feature_scaler.pkl"
ENCODER_ARTIFACT          = MODELS_DIR / "label_encoder.pkl"
INCIDENT_DB_PATH          = MODELS_DIR / "incident_history.json"

# ─────────────────────────────────────────────────────────────────────────────
# Attack label mapping  (attack_number column in merged CSV)
# ─────────────────────────────────────────────────────────────────────────────
ATTACK_LABELS = {
    0: "benign",
    1: "syn_flood",
    2: "udp_flood",
    3: "icmp_flood",
    4: "dns_flood",
    5: "gtp_u_flood",
}

ATTACK_LABELS_INV = {v: k for k, v in ATTACK_LABELS.items()}

ATTACK_SEVERITY = {
    "benign":      "none",
    "syn_flood":   "high",        # Targets AMF/SMF signaling plane
    "udp_flood":   "high",        # Saturates UPF data plane
    "icmp_flood":  "medium",      # Affects all network layers
    "dns_flood":   "medium-high", # Targets edge DNS
    "gtp_u_flood": "critical",    # Unique to 5G core – GTP-U tunnel flooding
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns (from dataset schema)
# ─────────────────────────────────────────────────────────────────────────────
UE_FEATURES = [
    "ul_bitrate",
    "dl_bitrate",
    "ul_retx",
    "dl_retx",
    "ul_mcs",
    "dl_mcs",
    "ul_phr",
    "cell_id",
    "pucch_snr",
    "pusch_snr",
]

ENB_FEATURES = [
    "enb_dl_bitrate",
    "enb_ul_bitrate",
    "enb_dl_usage_ratio",
    "enb_ul_usage_ratio",
    "gnb_cpu_load",
]

MME_FEATURES = [
    "pdu_session_setup_request",
    "pdu_session_setup_response",
    "paging_attempts",
    "context_setup_request",
    "context_setup_response",
    "abnormal_release_count",
]

ALL_FEATURES = UE_FEATURES + ENB_FEATURES + MME_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# LLM Backend Configuration
# ─────────────────────────────────────────────────────────────────────────────
LLMBackend = Literal["claude", "ollama", "openai_compatible"]

@dataclass
class LLMConfig:
    backend: LLMBackend = field(default_factory=lambda: os.getenv("LLM_BACKEND", "claude"))

    # ── Claude (Anthropic) ──────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    )

    # ── Ollama (local open-source LLMs) ─────────────────────────────────────
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    # Popular alternatives: mistral, phi4, gemma3, deepseek-r1, qwen2.5

    # ── OpenAI-compatible endpoint (vLLM, LM Studio, Groq, Together, etc.) ──
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # ── Common generation parameters ────────────────────────────────────────
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3"))
    )


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("MCP_PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    anomaly_threshold: float = field(
        default_factory=lambda: float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
    )


# Singletons loaded at import time (override by patching in tests)
llm_config    = LLMConfig()
server_config = ServerConfig()
