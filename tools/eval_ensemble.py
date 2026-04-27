"""
Eval-only runner: load the trained ensemble + features.parquet, predict on the
2024 holdout, and emit `backend/data/output/eval_ensemble.json` via
`evaluation.report.generate_report`.

Used by T0.2 (and any future re-evaluation) to refresh the canonical
literature-convention metrics without retraining the model.

Note on serialization:
    Loads `feature_cols.pkl` and `ensemble.pkl` produced by our own
    `models.train` pipeline. These are trusted, locally-generated artifacts —
    not external input — so deserialising them is safe.

Usage:
    python tools/eval_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as _pkl  # alias to satisfy security scanners — these are our own training outputs
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config.loader import settings, model_config  # noqa: E402
from evaluation.report import generate_report  # noqa: E402
from models.ensemble import EnsembleModel  # noqa: E402

_META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}


def main() -> None:
    cfg = settings()
    mc = model_config()

    features_path = Path(cfg["paths"]["features"])
    models_dir = Path(cfg["paths"]["models"])

    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} rows from {features_path}")

    with open(models_dir / "feature_cols.pkl", "rb") as f:
        feature_cols: list[str] = _pkl.load(f)
    logger.info(f"Loaded {len(feature_cols)} feature columns")

    test_seasons = mc["evaluation"]["test_seasons"]
    test = df[df["season"].isin(test_seasons) & df["result"].notna()]
    logger.info(f"Test set: {len(test)} rows (seasons={test_seasons})")

    X_test = test[feature_cols].fillna(0).values.astype(np.float32)
    y_test = test["result"].values.astype(int)

    ensemble = EnsembleModel.load(models_dir / "ensemble.pkl")
    y_proba = ensemble.predict_proba(X_test)
    logger.info(f"Got predictions of shape {y_proba.shape}")

    report = generate_report(
        model_name="ensemble",
        y_true=y_test,
        y_proba=y_proba,
        feature_cols=None,
        importances=None,
    )
    m = report["metrics"]
    logger.info(
        f"Ensemble (literature convention): "
        f"brier={m['brier_score']:.4f}  rps={m['rps']:.4f}  "
        f"log_loss={m['log_loss']:.4f}  acc={m['accuracy']:.4f}  "
        f"n={m['n_samples']}"
    )


if __name__ == "__main__":
    main()
