#!/usr/bin/env python3
"""
Train XGBoost classifiers on the NCSRD-DS-5GDDoS dataset.

Prerequisites:
    1. Download the dataset (see data/DATASET_INSTRUCTIONS.md)
    2. Install dependencies: pip install -r requirements.txt

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --nrows 100000   # Quick run on 100k rows
    python scripts/train_model.py --no-balance      # Skip SMOTE-Tomek

Output:
    models/xgb_binary_classifier.pkl
    models/xgb_multiclass_classifier.pkl
    models/feature_scaler.pkl
    models/training_report.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train 5G-DDoS classifiers")
    parser.add_argument("--nrows",      type=int,  default=None, help="Limit rows loaded (for testing)")
    parser.add_argument("--no-balance", action="store_true",     help="Skip SMOTE-Tomek class balancing")
    parser.add_argument("--test-frac",  type=float, default=0.2, help="Test set fraction (default: 0.2)")
    args = parser.parse_args()

    # ── Imports ───────────────────────────────────────────────────────────
    try:
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import (
            accuracy_score, classification_report, f1_score, roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        logger.error("Missing dependency: %s\nRun: pip install -r requirements.txt", exc)
        sys.exit(1)

    from src.config import ATTACK_LABELS, MODELS_DIR
    from src.data.loader import load_merged
    from src.data.preprocessor import (
        balance_classes,
        build_feature_matrix,
        normalise_columns,
        train_test_split_temporal,
    )
    from src.models.classifier import FivGDDoSClassifier

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading merged dataset (nrows=%s)…", args.nrows)
    t0 = time.time()
    df = load_merged(nrows=args.nrows)
    logger.info("Loaded in %.1fs – %d rows, %d cols", time.time() - t0, len(df), len(df.columns))

    df = normalise_columns(df)

    # ── Class distribution ────────────────────────────────────────────────
    label_col = "attack_number"
    if label_col not in df.columns:
        # Try to find the label column
        candidates = [c for c in df.columns if "attack" in c.lower()]
        if candidates:
            label_col = candidates[0]
            logger.info("Using label column: %s", label_col)
        else:
            logger.error("Label column 'attack_number' not found. Columns: %s", list(df.columns[:20]))
            sys.exit(1)

    logger.info("Class distribution:\n%s", df[label_col].value_counts().to_string())

    # ── Train / test split (temporal – no shuffle to prevent leakage) ─────
    train_df, test_df = train_test_split_temporal(df, test_fraction=args.test_frac)
    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    X_train, y_train_multi = build_feature_matrix(train_df, label_col=label_col, binary=False)
    X_test,  y_test_multi  = build_feature_matrix(test_df,  label_col=label_col, binary=False)
    y_train_bin = (y_train_multi > 0).astype(int)
    y_test_bin  = (y_test_multi  > 0).astype(int)

    # ── Feature scaling ───────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Class balancing (SMOTE-Tomek – proven on this dataset) ───────────
    if not args.no_balance:
        logger.info("Applying SMOTE-Tomek class balancing…")
        try:
            X_train_bal, y_train_bal_multi = balance_classes(X_train, y_train_multi, "smote_tomek")
            X_train_bin_bal = X_train_bal
            y_train_bin_bal = (y_train_bal_multi > 0).astype(int)
        except ImportError:
            logger.warning("imbalanced-learn not installed – skipping class balancing")
            X_train_bin_bal, y_train_bin_bal = X_train, y_train_bin
            X_train_bal,     y_train_bal_multi = X_train, y_train_multi
    else:
        X_train_bin_bal, y_train_bin_bal = X_train, y_train_bin
        X_train_bal,     y_train_bal_multi = X_train, y_train_multi

    # ── Binary classifier ─────────────────────────────────────────────────
    logger.info("Training binary XGBoost classifier…")
    t1 = time.time()
    binary_clf = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        eval_metric       = "logloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )
    binary_clf.fit(X_train_bin_bal, y_train_bin_bal)
    logger.info("Binary model trained in %.1fs", time.time() - t1)

    y_pred_bin   = binary_clf.predict(X_test)
    y_proba_bin  = binary_clf.predict_proba(X_test)[:, 1]
    bin_f1       = f1_score(y_test_bin, y_pred_bin)
    bin_auc      = roc_auc_score(y_test_bin, y_proba_bin)
    logger.info("Binary  → F1: %.4f | AUC-ROC: %.4f", bin_f1, bin_auc)
    logger.info("\n%s", classification_report(y_test_bin, y_pred_bin, target_names=["benign", "attack"]))

    # ── Multi-class classifier ────────────────────────────────────────────
    logger.info("Training multi-class XGBoost classifier (6 classes)…")
    t2 = time.time()
    n_classes   = len(ATTACK_LABELS)
    multi_clf   = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        objective         = "multi:softprob",
        num_class         = n_classes,
        eval_metric       = "mlogloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )
    multi_clf.fit(X_train_bal, y_train_bal_multi)
    logger.info("Multi-class model trained in %.1fs", time.time() - t2)

    y_pred_multi = multi_clf.predict(X_test)
    mc_f1 = f1_score(y_test_multi, y_pred_multi, average="weighted")
    target_names = [ATTACK_LABELS[i] for i in range(n_classes)]
    logger.info("Multi-class → Weighted F1: %.4f", mc_f1)
    logger.info("\n%s", classification_report(y_test_multi, y_pred_multi, target_names=target_names))

    # ── Save artifacts ────────────────────────────────────────────────────
    clf_wrapper = FivGDDoSClassifier()
    clf_wrapper.save(binary_clf, multi_clf, scaler)

    # Save training report
    report = {
        "dataset":          "NCSRD-DS-5GDDoS v3.0",
        "train_rows":       int(len(X_train_bin_bal)),
        "test_rows":        int(len(X_test)),
        "binary_f1":        round(bin_f1, 4),
        "binary_auc_roc":   round(bin_auc, 4),
        "multiclass_f1":    round(mc_f1, 4),
        "class_balancing":  "SMOTE-Tomek" if not args.no_balance else "none",
        "model":            "XGBoost",
    }
    report_path = MODELS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n✅ Training complete!")
    logger.info("   Binary  F1   : %.4f  (target: >0.95)", bin_f1)
    logger.info("   Binary  AUC  : %.4f  (target: >0.99)", bin_auc)
    logger.info("   MultiCls F1  : %.4f  (target: >0.95)", mc_f1)
    logger.info("   Artifacts    : %s", MODELS_DIR)
    logger.info("   Report       : %s", report_path)


if __name__ == "__main__":
    main()
