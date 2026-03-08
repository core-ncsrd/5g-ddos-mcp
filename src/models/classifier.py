"""
5G DDoS Attack Classifier

Wraps an XGBoost (or scikit-learn) model trained on the NCSRD-DS-5GDDoS dataset.
Supports:
  - Binary classification  : benign (0) vs. attack (1)
  - Multi-class            : benign / syn_flood / udp_flood / icmp_flood /
                             dns_flood / gtp_u_flood  (attack_number 0-5)

The trained model artifacts are NOT included in the repo.
Run `scripts/train_model.py` after downloading the dataset to generate them.
See data/DATASET_INSTRUCTIONS.md for download instructions.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    ALL_FEATURES,
    ATTACK_LABELS,
    MODELS_DIR,
    MODEL_ARTIFACT_BINARY,
    MODEL_ARTIFACT_MULTICLASS,
    SCALER_ARTIFACT,
)

logger = logging.getLogger(__name__)


class FivGDDoSClassifier:
    """
    Thin wrapper around trained sklearn / XGBoost estimators.

    Lifecycle:
        clf = FivGDDoSClassifier()
        clf.load()                        # loads artifacts from models/
        result = clf.predict(features)    # dict with prediction + probabilities
    """

    def __init__(self):
        self._binary_model:     Optional[Any] = None
        self._multiclass_model: Optional[Any] = None
        self._scaler:           Optional[Any] = None
        self._loaded: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Load / save artifacts
    # ──────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load pre-trained model artifacts. Call once at startup."""
        if not MODEL_ARTIFACT_BINARY.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_ARTIFACT_BINARY}.\n"
                "Train the model first: python scripts/train_model.py\n"
                "Dataset must be downloaded first: see data/DATASET_INSTRUCTIONS.md"
            )

        with open(MODEL_ARTIFACT_BINARY, "rb") as f:
            self._binary_model = pickle.load(f)
        logger.info("Loaded binary classifier from %s", MODEL_ARTIFACT_BINARY)

        if MODEL_ARTIFACT_MULTICLASS.exists():
            with open(MODEL_ARTIFACT_MULTICLASS, "rb") as f:
                self._multiclass_model = pickle.load(f)
            logger.info("Loaded multi-class classifier from %s", MODEL_ARTIFACT_MULTICLASS)

        if SCALER_ARTIFACT.exists():
            with open(SCALER_ARTIFACT, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("Loaded feature scaler from %s", SCALER_ARTIFACT)

        self._loaded = True

    def save(self, binary_model, multiclass_model, scaler=None) -> None:
        """Persist trained models to disk (called from train_model.py)."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_ARTIFACT_BINARY, "wb") as f:
            pickle.dump(binary_model, f)
        with open(MODEL_ARTIFACT_MULTICLASS, "wb") as f:
            pickle.dump(multiclass_model, f)
        if scaler is not None:
            with open(SCALER_ARTIFACT, "wb") as f:
                pickle.dump(scaler, f)
        logger.info("Saved model artifacts to %s", MODELS_DIR)

    # ──────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict attack type from a telemetry feature dict.

        Parameters
        ----------
        features : dict mapping feature name → float value
            Required keys: see src.config.ALL_FEATURES
            Missing features are filled with 0.0

        Returns
        -------
        dict with keys:
            is_attack          : bool
            attack_type        : str  (e.g. "gtp_u_flood")
            attack_number      : int  (0-5)
            binary_probability : float  (probability of being an attack)
            class_probabilities: dict   {attack_label: probability}
            severity           : str    ("none" / "medium" / "high" / "critical")
            features_used      : list[str]
        """
        if not self._loaded:
            raise RuntimeError("Call classifier.load() before predict()")

        x = self._build_feature_vector(features)

        # ── Binary prediction ─────────────────────────────────────────
        binary_proba = float(self._binary_model.predict_proba(x)[0][1])
        is_attack    = binary_proba >= 0.5

        # ── Multi-class prediction ────────────────────────────────────
        attack_number = 0
        class_probs   = {lbl: 0.0 for lbl in ATTACK_LABELS.values()}

        if self._multiclass_model is not None:
            mc_proba    = self._multiclass_model.predict_proba(x)[0]
            attack_number = int(np.argmax(mc_proba))
            for idx, prob in enumerate(mc_proba):
                label = ATTACK_LABELS.get(idx, f"class_{idx}")
                class_probs[label] = round(float(prob), 4)
        elif is_attack:
            # Fallback: use binary model only
            attack_number = 1  # unknown attack type

        attack_type = ATTACK_LABELS.get(attack_number, "unknown")

        from src.config import ATTACK_SEVERITY
        severity = ATTACK_SEVERITY.get(attack_type, "unknown")

        return {
            "is_attack":           is_attack,
            "attack_type":         attack_type,
            "attack_number":       attack_number,
            "binary_probability":  round(binary_proba, 4),
            "class_probabilities": class_probs,
            "severity":            severity,
            "features_used":       list(features.keys()),
        }

    def predict_batch(
        self, feature_rows: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Predict multiple telemetry snapshots at once."""
        return [self.predict(row) for row in feature_rows]

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Fill missing features with 0 and return (1, n_features) array."""
        vec = np.array(
            [float(features.get(f, 0.0)) for f in ALL_FEATURES],
            dtype=np.float32,
        ).reshape(1, -1)

        if self._scaler is not None:
            vec = self._scaler.transform(vec)

        return vec

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def feature_names(self) -> List[str]:
        return list(ALL_FEATURES)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton  (lazy-loaded on first use)
# ──────────────────────────────────────────────────────────────────────────────
_classifier: Optional[FivGDDoSClassifier] = None


def get_classifier() -> FivGDDoSClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FivGDDoSClassifier()
        try:
            _classifier.load()
        except FileNotFoundError as exc:
            logger.warning(
                "Model artifacts not found – classifier running in demo mode.\n%s", exc
            )
    return _classifier
