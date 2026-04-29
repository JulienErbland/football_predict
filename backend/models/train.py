"""
Training orchestrator — walk-forward CV + locked holdout + quality gates.

Pipeline (T2.1, supersedes the placeholder Phase 1 path):
    1. Load features.parquet.
    2. Walk-forward CV across 2021/2022/2023 (six folds at the locked
       (n_splits=2, vw=9) parametrization from commit 2). Each fold trains
       calibrated XGB + LightGBM, builds a weighted-average ensemble,
       evaluates on the val window.
    3. Final retrain on all CV-pool data → ensemble.pkl.
    4. Holdout evaluation on 2024 — :class:`HoldoutSection`.
    5. Compute quality gates → :class:`GatesSection`.
    6. Save artifacts (ensemble.pkl, feature_cols.pkl, eval_ensemble.json).
    7. Tier 1 — call ``cv_report.assert_gates()``. Failure ⇒ non-zero exit
       (CI blocks merge); artifacts persist for inspection.

Expected first-run failure: Phase 1's draw_f1 baseline of ~0.086 sits well
below the ``min_draw_f1=0.25`` gate. This is correct — T2.1 ships the
gate, not a model that passes it. T2.2 (SMOTE + class weights + threshold
calibration) is the green-run ticket.

CLI usage:
    cd backend
    python -m ingestion.football_data_csv --leagues PL,PD,BL1,SA,FL1 --seasons 2021,2022,2023,2024
    python -m features.build
    python -m models.train
"""

from __future__ import annotations

import argparse
import json
import pickle  # trusted local file pattern, mirrors models/ensemble.py
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, recall_score

from config.loader import settings, model_config
from config.schema import load_model_config
from evaluation.cv import train_calibrated_models, run_cv
from evaluation.cv_report import (
    CalibrationSection, CVReport, FoldMetrics, GatesSection, HoldoutSection,
    SCHEMA_VERSION,
)
from evaluation.exceptions import HoldoutSnapshotMismatch, QualityGateFailure
from evaluation.metrics import evaluate_predictions
from evaluation.splits import WalkForwardSplit
from features.build import FEATURE_SCHEMA_VERSION
from models.ensemble import EnsembleModel
from tools.bootstrap_holdout_snapshot import hash_match_ids


_META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {path}. Run features.build first."
        )
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _retrain_final(
    pool_df: pd.DataFrame, feature_cols: list[str], mc: dict,
) -> EnsembleModel:
    """Retrain on the full CV pool. Returns the calibrated ensemble."""
    logger.info(f"Retraining final models on {len(pool_df)} CV-pool rows...")
    X_all = pool_df[feature_cols].fillna(0).to_numpy(np.float32)
    y_all = pool_df["result"].to_numpy(int)
    cut = int(len(X_all) * 0.9)
    X_tr, y_tr = X_all[:cut], y_all[:cut]
    X_val, y_val = X_all[cut:], y_all[cut:]

    models = train_calibrated_models(X_tr, y_tr, X_val, y_val, feature_cols, mc)
    weights = {name: mc["models"][name]["weight"] for name in models}
    return EnsembleModel(models, weights)


def _evaluate_holdout(
    holdout_df: pd.DataFrame, ensemble: EnsembleModel,
    feature_cols: list[str], snapshot_hash: str,
) -> HoldoutSection:
    X = holdout_df[feature_cols].fillna(0).to_numpy(np.float32)
    y = holdout_df["result"].to_numpy(int)
    proba = ensemble.predict_proba(X)
    proba = np.clip(proba, 0.0, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)

    base = evaluate_predictions(y, proba)
    pred = proba.argmax(axis=1)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    metrics = FoldMetrics(
        brier=base["brier_score"],
        rps=base["rps"],
        log_loss=base["log_loss"],
        accuracy=base["accuracy"],
        draw_f1=float(f1_score(y, pred, labels=[1], average="macro", zero_division=0)),
        home_recall=float(rec[0]), draw_recall=float(rec[1]), away_recall=float(rec[2]),
    )
    return HoldoutSection(
        season=int(holdout_df["season"].iloc[0]),
        n_test=int(len(holdout_df)),
        metrics=metrics,
        snapshot_hash=snapshot_hash,
    )


def _build_gates(
    cv_mean: FoldMetrics, holdout_metrics: FoldMetrics, training_cfg,
) -> GatesSection:
    failures: list[str] = []
    if cv_mean.rps > training_cfg.max_rps:
        failures.append(f"cv_mean_rps {cv_mean.rps:.4f} > {training_cfg.max_rps}")
    if cv_mean.brier > training_cfg.max_brier:
        failures.append(f"cv_mean_brier {cv_mean.brier:.4f} > {training_cfg.max_brier}")
    if cv_mean.draw_f1 < training_cfg.min_draw_f1:
        failures.append(f"cv_mean_draw_f1 {cv_mean.draw_f1:.4f} < {training_cfg.min_draw_f1}")
    if holdout_metrics.rps > training_cfg.max_rps:
        failures.append(f"holdout_rps {holdout_metrics.rps:.4f} > {training_cfg.max_rps}")
    if holdout_metrics.brier > training_cfg.max_brier:
        failures.append(f"holdout_brier {holdout_metrics.brier:.4f} > {training_cfg.max_brier}")
    if holdout_metrics.draw_f1 < training_cfg.min_draw_f1:
        failures.append(f"holdout_draw_f1 {holdout_metrics.draw_f1:.4f} < {training_cfg.min_draw_f1}")

    return GatesSection(
        max_rps=training_cfg.max_rps,
        max_brier=training_cfg.max_brier,
        min_draw_f1=training_cfg.min_draw_f1,
        cv_mean_rps=cv_mean.rps,
        cv_mean_brier=cv_mean.brier,
        cv_mean_draw_f1=cv_mean.draw_f1,
        holdout_rps=holdout_metrics.rps,
        holdout_brier=holdout_metrics.brier,
        holdout_draw_f1=holdout_metrics.draw_f1,
        passed=len(failures) == 0,
        failures=tuple(failures),
    )


def _feature_importances(
    ensemble: EnsembleModel, feature_cols: list[str],
) -> dict[str, float]:
    """Top-50 feature importances from the XGBoost member of the ensemble."""
    xgb = ensemble.models.get("xgboost")
    if xgb is None:
        return {}
    importances = getattr(xgb, "feature_importances_", None)
    if importances is None:
        base = getattr(xgb, "model", None) or getattr(xgb, "_base", None)
        importances = getattr(base, "feature_importances_", None)
    if importances is None:
        return {}
    pairs = sorted(
        zip(feature_cols, [float(v) for v in importances]),
        key=lambda kv: abs(kv[1]), reverse=True,
    )
    return dict(pairs[:50])


def _verify_holdout_snapshot(models_dir: Path, holdout_df: pd.DataFrame) -> str:
    """Verify the on-disk snapshot matches the live holdout match_ids.

    Returns the snapshot hash on success. Raises HoldoutSnapshotMismatch
    on drift, surfacing the symmetric difference so operators can debug
    whether ingestion added/removed matches or the wrong season was
    sampled.
    """
    path = models_dir / "holdout_snapshot.v1.json"
    if not path.exists():
        logger.warning(
            f"{path.name} not found — proceeding without snapshot verification. "
            "Run `python -m tools.bootstrap_holdout_snapshot` to seal."
        )
        return "sha256:not-yet-sealed"

    with open(path) as f:
        snap = json.load(f)

    snap_ids = set(snap.get("match_ids", []))
    live_ids = set(str(mid) for mid in holdout_df["match_id"].tolist())
    if snap_ids != live_ids:
        only_snap = sorted(snap_ids - live_ids)[:5]
        only_live = sorted(live_ids - snap_ids)[:5]
        raise HoldoutSnapshotMismatch(
            f"Holdout match_ids drifted from {path.name}. "
            f"Snapshot hash={snap.get('match_ids_sha256')}, "
            f"live hash={hash_match_ids(sorted(live_ids))}. "
            f"In snapshot but not live (first 5): {only_snap}. "
            f"In live but not snapshot (first 5): {only_live}."
        )
    return snap["match_ids_sha256"]


def train(force_retrain: bool = False) -> EnsembleModel:
    cfg = settings()
    mc = model_config()
    training_cfg = load_model_config().training

    features_path = Path(cfg["paths"]["features"])
    models_dir = Path(cfg["paths"]["models"])
    output_dir = Path(cfg["paths"]["output"])
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_features(features_path)
    feature_cols = _feature_cols(df)
    logger.info(f"Feature columns: {len(feature_cols)} (schema {FEATURE_SCHEMA_VERSION})")

    splitter = WalkForwardSplit()
    weights = {
        name: mc["models"][name]["weight"]
        for name in ("xgboost", "lightgbm")
        if mc["models"][name]["enabled"]
    }
    if not weights:
        raise RuntimeError("No models enabled in model_config.yaml.")

    logger.info(
        f"Walk-forward CV: pool={splitter.cv_pool_seasons} "
        f"holdout={splitter.holdout_season} "
        f"({splitter.expected_n_folds} folds)"
    )
    cv_section = run_cv(df, feature_cols, splitter, mc, weights)
    logger.info(
        f"CV mean: rps={cv_section.mean_metrics.rps:.4f} "
        f"brier={cv_section.mean_metrics.brier:.4f} "
        f"draw_f1={cv_section.mean_metrics.draw_f1:.4f}"
    )

    pool_df = df[df["season"].isin(splitter.cv_pool_seasons)].sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)
    holdout_df = df[df["season"] == splitter.holdout_season].sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    final_ensemble = _retrain_final(pool_df, feature_cols, mc)
    snapshot_hash = _verify_holdout_snapshot(models_dir, holdout_df)
    holdout_section = _evaluate_holdout(holdout_df, final_ensemble, feature_cols, snapshot_hash)
    logger.info(
        f"Holdout (n={holdout_section.n_test}): "
        f"rps={holdout_section.metrics.rps:.4f} "
        f"brier={holdout_section.metrics.brier:.4f} "
        f"draw_f1={holdout_section.metrics.draw_f1:.4f}"
    )

    # Save artifacts BEFORE asserting gates — operators need them to debug.
    ensemble_path = models_dir / "ensemble.pkl"
    final_ensemble.save(ensemble_path)
    feature_cols_path = models_dir / "feature_cols.pkl"
    with open(feature_cols_path, "wb") as f:
        pickle.dump(feature_cols, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved {ensemble_path} and {feature_cols_path}")

    gates = _build_gates(cv_section.mean_metrics, holdout_section.metrics, training_cfg)
    cv_report = CVReport(
        schema_version=SCHEMA_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cv=cv_section,
        holdout=holdout_section,
        gates=gates,
        calibration=CalibrationSection(method="isotonic", cv_folds=splitter.expected_n_folds),
        feature_importances=_feature_importances(final_ensemble, feature_cols),
    )
    eval_path = output_dir / "eval_ensemble.json"
    eval_path.write_text(cv_report.to_json())
    logger.info(f"Saved {eval_path}")

    # Tier 1 — gate enforcement. Non-zero exit on failure.
    try:
        cv_report.assert_gates()
    except QualityGateFailure as e:
        logger.error(e.verbose_breakdown())
        raise

    logger.info("Training complete; gates passed.")
    return final_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Train football prediction models")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if models already exist")
    args = parser.parse_args()
    train(force_retrain=args.force)


if __name__ == "__main__":
    main()
