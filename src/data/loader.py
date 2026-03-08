"""
Dataset loader for the NCSRD-DS-5GDDoS dataset.

Dataset must be downloaded manually from Zenodo before use.
See: data/DATASET_INSTRUCTIONS.md

DOI: 10.5281/zenodo.13900057
License: CC BY 4.0
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import DATA_DIR, DATASET_FILES

logger = logging.getLogger(__name__)


def _csv_path(key: str) -> Path:
    return DATA_DIR / DATASET_FILES[key]


def check_dataset_available() -> dict:
    """
    Return availability status for each dataset file.
    Does NOT download anything.
    """
    status = {}
    for key, filename in DATASET_FILES.items():
        path = DATA_DIR / filename
        status[key] = {
            "filename": filename,
            "path":     str(path),
            "exists":   path.exists(),
            "size_mb":  round(path.stat().st_size / 1e6, 1) if path.exists() else None,
        }
    return status


def load_merged(
    nrows: Optional[int] = None,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load the primary ML-ready file:
        amari_ue_data_merged_with_attack_number.csv  (241.5 MB)

    This file contains both benign and malicious UE records with
    the `attack_number` label column (0=benign, 1-5=attack types).

    Parameters
    ----------
    nrows   : Load only first N rows (useful for EDA / testing).
    columns : Subset of columns to load.
    """
    path = _csv_path("merged")
    _assert_exists(path, "merged")

    logger.info("Loading merged dataset from %s (nrows=%s)…", path, nrows)
    df = pd.read_csv(path, nrows=nrows, usecols=columns, low_memory=False)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def load_ue_classic(nrows: Optional[int] = None) -> pd.DataFrame:
    """UE metrics from Amarisoft Classic cells (143.6 MB)."""
    path = _csv_path("ue_classic")
    _assert_exists(path, "ue_classic")
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_ue_mini(nrows: Optional[int] = None) -> pd.DataFrame:
    """UE metrics from Amarisoft Mini cell 2 (87.3 MB)."""
    path = _csv_path("ue_mini")
    _assert_exists(path, "ue_mini")
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_enb_classic(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Cell-level eNB counters from Classic cells (72.2 MB).
    Columns include: enb_dl_bitrate, enb_ul_bitrate, gnb_cpu_load, …
    """
    path = _csv_path("enb_classic")
    _assert_exists(path, "enb_classic")
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_enb_mini(nrows: Optional[int] = None) -> pd.DataFrame:
    """Cell-level eNB counters from Mini cell (38.2 MB)."""
    path = _csv_path("enb_mini")
    _assert_exists(path, "enb_mini")
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_mme_counters(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    NAS-layer MME counters (37.1 MB).
    Columns include: pdu_session_setup_*, paging_attempts,
                     abnormal_release_count, …
    """
    path = _csv_path("mme")
    _assert_exists(path, "mme")
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_summary_report() -> pd.DataFrame:
    """
    Attack summary spreadsheet (18 kB).
    Maps IMEISV / IP / device type to attack type per malicious UE.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("Install openpyxl: pip install openpyxl")
    path = _csv_path("summary")
    _assert_exists(path, "summary")
    return pd.read_excel(path)


def load_cross_layer_features(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a holistic per-UE feature vector by joining:
        UE telemetry  +  eNB counters  +  MME NAS counters

    Join key: timestamp (rounded to the nearest second).
    Missing counters are filled with 0.

    Returns the merged DataFrame ready for feature engineering.
    """
    logger.info("Building cross-layer feature set…")
    df_ue  = load_merged(nrows=nrows)
    df_enb = _load_enb_combined(nrows=nrows)
    df_mme = load_mme_counters(nrows=nrows)

    # Normalise timestamp column name
    for df in (df_ue, df_enb, df_mme):
        if "timestamp" not in df.columns:
            ts_cols = [c for c in df.columns if "time" in c.lower()]
            if ts_cols:
                df.rename(columns={ts_cols[0]: "timestamp"}, inplace=True)

    df_ue["timestamp"]  = pd.to_datetime(df_ue["timestamp"],  errors="coerce")
    df_enb["timestamp"] = pd.to_datetime(df_enb["timestamp"], errors="coerce")
    df_mme["timestamp"] = pd.to_datetime(df_mme["timestamp"], errors="coerce")

    # Round to nearest second for join tolerance
    for df in (df_ue, df_enb, df_mme):
        df["ts_sec"] = df["timestamp"].dt.floor("s")

    merged = df_ue.merge(df_enb, on="ts_sec", how="left", suffixes=("", "_enb"))
    merged = merged.merge(df_mme, on="ts_sec", how="left", suffixes=("", "_mme"))
    merged.fillna(0, inplace=True)

    logger.info("Cross-layer dataset: %d rows × %d columns", len(merged), len(merged.columns))
    return merged


def _load_enb_combined(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load and concatenate Classic + Mini eNB counters."""
    dfs = []
    for key in ("enb_classic", "enb_mini"):
        path = _csv_path(key)
        if path.exists():
            dfs.append(pd.read_csv(path, nrows=nrows, low_memory=False))
    if not dfs:
        raise FileNotFoundError(
            "eNB counter files not found. See data/DATASET_INSTRUCTIONS.md"
        )
    return pd.concat(dfs, ignore_index=True)


def _assert_exists(path: Path, key: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file '{path.name}' not found at {path}.\n"
            f"Download the NCSRD-DS-5GDDoS dataset from Zenodo:\n"
            f"  DOI:  https://doi.org/10.5281/zenodo.13900057\n"
            f"  Place the file in the ./data/ directory.\n"
            f"  Full instructions: data/DATASET_INSTRUCTIONS.md"
        )
