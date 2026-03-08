"""
Feature engineering + preprocessing pipeline for the NCSRD-DS-5GDDoS dataset.

Key transformations:
  1. Column normalisation (map raw CSV column names to canonical names)
  2. Temporal windowing  (sliding window aggregation for time-series)
  3. SMOTE-Tomek class balancing  (training only)
  4. Feature scaling  (StandardScaler saved to models/)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, ATTACK_LABELS_INV, ENB_FEATURES, MME_FEATURES, UE_FEATURES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Column name aliases  (dataset CSV headers → canonical names)
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    # UE-level
    "ul_bitrate":           ["ul_bitrate", "ul_bit_rate", "uplink_bitrate"],
    "dl_bitrate":           ["dl_bitrate", "dl_bit_rate", "downlink_bitrate"],
    "ul_retx":              ["ul_retx", "ul_retransmissions", "ul_retx_count"],
    "dl_retx":              ["dl_retx", "dl_retransmissions", "dl_retx_count"],
    "ul_mcs":               ["ul_mcs", "ul_mod_coding_scheme"],
    "dl_mcs":               ["dl_mcs", "dl_mod_coding_scheme"],
    "ul_phr":               ["ul_phr", "power_headroom"],
    "cell_id":              ["cell_id", "cellid", "cell"],
    "pucch_snr":            ["pucch_snr", "pucch_snr_db"],
    "pusch_snr":            ["pusch_snr", "pusch_snr_db"],
    # eNB-level
    "enb_dl_bitrate":       ["enb_dl_bitrate", "enb_dl_bit_rate"],
    "enb_ul_bitrate":       ["enb_ul_bitrate", "enb_ul_bit_rate"],
    "enb_dl_usage_ratio":   ["enb_dl_usage_ratio", "dl_usage_ratio"],
    "enb_ul_usage_ratio":   ["enb_ul_usage_ratio", "ul_usage_ratio"],
    "gnb_cpu_load":         ["gnb_cpu_load", "cpu_load", "gnb_cpu"],
    # MME / NAS
    "pdu_session_setup_request":  ["pdu_session_setup_request", "pdu_setup_req"],
    "pdu_session_setup_response": ["pdu_session_setup_response", "pdu_setup_resp"],
    "paging_attempts":            ["paging_attempts", "paging_count"],
    "context_setup_request":      ["context_setup_request", "ctx_setup_req"],
    "context_setup_response":     ["context_setup_response", "ctx_setup_resp"],
    "abnormal_release_count":     ["abnormal_release_count", "abnormal_releases"],
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV columns to canonical names using COLUMN_ALIASES."""
    rename_map: dict = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in cols_lower:
                rename_map[cols_lower[alias.lower()]] = canonical
                break
    if rename_map:
        logger.debug("Renaming columns: %s", rename_map)
        df = df.rename(columns=rename_map)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and fill ALL_FEATURES from a normalised DataFrame.
    Missing columns are filled with 0.
    """
    df = normalise_columns(df)
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        logger.warning("Missing features (filling with 0): %s", missing)
        for col in missing:
            df[col] = 0.0
    return df[ALL_FEATURES].astype(np.float32)


def apply_temporal_windows(
    df: pd.DataFrame,
    window_sizes: Tuple[int, ...] = (10, 30, 60),
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add sliding window aggregation columns for bitrate and counter features.
    Window sizes are in seconds.

    Critical for detecting GTP-U and SYN flood patterns which only become
    apparent over a time window.
    """
    if timestamp_col not in df.columns:
        logger.warning("No '%s' column – skipping temporal windowing", timestamp_col)
        return df

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.sort_values(timestamp_col)

    numeric_cols = [
        "ul_bitrate", "dl_bitrate", "ul_retx",
        "enb_dl_bitrate", "gnb_cpu_load",
        "pdu_session_setup_request", "abnormal_release_count",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    df = df.set_index(timestamp_col)
    for w in window_sizes:
        rule = f"{w}s"
        for col in numeric_cols:
            df[f"{col}_mean_{w}s"] = df[col].rolling(rule).mean()
            df[f"{col}_std_{w}s"]  = df[col].rolling(rule).std().fillna(0)
    df = df.reset_index()
    return df


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote_tomek",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution using SMOTE-Tomek (proven on this dataset).
    Only call on training data – never on validation/test splits.

    Requires: pip install imbalanced-learn
    """
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "Install imbalanced-learn: pip install imbalanced-learn"
        )

    original_counts = dict(zip(*np.unique(y, return_counts=True)))
    logger.info("Class distribution before balancing: %s", original_counts)

    if strategy == "smote_tomek":
        sampler = SMOTETomek(random_state=42)
    elif strategy == "smote":
        sampler = SMOTE(random_state=42)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    X_res, y_res = sampler.fit_resample(X, y)
    balanced_counts = dict(zip(*np.unique(y_res, return_counts=True)))
    logger.info("Class distribution after  balancing: %s", balanced_counts)
    return X_res, y_res


def train_test_split_temporal(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split (no shuffle) to prevent data leakage.
    The last `test_fraction` of records by timestamp form the test set.
    """
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_feature_matrix(
    df: pd.DataFrame,
    label_col: str = "attack_number",
    binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) ready for scikit-learn / XGBoost training.

    Parameters
    ----------
    df        : Normalised DataFrame containing ALL_FEATURES + label_col
    label_col : Column containing attack_number (0-5)
    binary    : If True, collapse labels to 0 (benign) / 1 (attack)
    """
    X = extract_features(df).values
    y = df[label_col].values.astype(int)
    if binary:
        y = (y > 0).astype(int)
    return X, y
