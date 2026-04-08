"""
Evaluation report generator.

Produces a JSON report for each model containing all metrics, calibration stats,
and feature importances. Reports are saved to data/output/ and can be linked
from the frontend for model transparency.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

from config.loader import settings
from evaluation.metrics import evaluate_predictions, calibration_summary


def generate_report(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    feature_cols: list[str] | None = None,
    importances: np.ndarray | None = None,
) -> dict:
    """
    Generate and save a JSON evaluation report for a model.

    Args:
        model_name: identifier string (e.g. "xgboost", "ensemble")
        y_true: ground truth class labels (0/1/2)
        y_proba: predicted probabilities, shape (n, 3)
        feature_cols: list of feature names (for importance reporting)
        importances: feature importance array (same length as feature_cols)

    Returns the report dict (also saved to data/output/eval_{model_name}.json).
    """
    cfg = settings()
    out_dir = Path(cfg["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_predictions(y_true, y_proba)
    cal_stats = calibration_summary(y_true, y_proba)

    report: dict = {
        "model_name": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "calibration": cal_stats,
        "feature_importances": [],
    }

    if feature_cols is not None and importances is not None:
        sorted_indices = np.argsort(importances)[::-1]
        report["feature_importances"] = [
            {"feature": feature_cols[i], "importance": float(importances[i])}
            for i in sorted_indices[:50]  # Top 50 features
        ]

    out_path = out_dir / f"eval_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        f"Saved evaluation report for {model_name} → {out_path} "
        f"(brier={metrics['brier_score']:.4f}, rps={metrics['rps']:.4f})"
    )
    return report
