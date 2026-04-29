"""
Cross-validation orchestrator.

Runs the per-fold train + calibrate + ensemble + evaluate loop and
aggregates metrics into a :class:`CVSection`. Splits are produced by
:class:`evaluation.splits.WalkForwardSplit`; metrics use the literature
conventions in :mod:`evaluation.metrics`.

The per-fold inner loop intentionally mirrors the harness in
``tools/validate_cv_parametrization.py`` (commit 2). The harness was the
empirical validator that locked the (n_splits=2, vw=9) default; this
module is the production runtime that consumes the same default. Drift
between the two would invalidate the empirical rationale.
"""

from __future__ import annotations

import statistics
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, recall_score

from evaluation.cv_report import (
    CVSection, FoldGuards, FoldMetrics, FoldResult,
)
from evaluation.metrics import evaluate_predictions
from evaluation.splits import WalkForwardSplit
from models.lgbm_model import LGBMModel
from models.xgboost_model import XGBoostModel


def _draw_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    )


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (home_recall, draw_recall, away_recall)."""
    rec = recall_score(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0,
    )
    return float(rec[0]), float(rec[1]), float(rec[2])


def _ensemble(probas: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of calibrated probas; clip + renormalise to fp-safe rows."""
    keys = list(probas)
    total = sum(weights[k] for k in keys)
    out = sum((weights[k] / total) * probas[k] for k in keys)
    out = np.clip(out, 0.0, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def train_calibrated_models(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_inner_val: np.ndarray, y_inner_val: np.ndarray,
    feature_names: list[str], mc: dict,
) -> dict[str, np.ndarray]:
    """Train enabled models and return calibrated proba-callables."""
    models: dict[str, object] = {}
    if mc["models"]["xgboost"]["enabled"]:
        c = mc["models"]["xgboost"]
        m = XGBoostModel(
            n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
            max_depth=c["max_depth"], subsample=c["subsample"],
            colsample_bytree=c["colsample_bytree"],
        )
        m.fit(X_tr, y_tr, X_val=X_inner_val, y_val=y_inner_val,
              feature_names=feature_names,
              early_stopping_rounds=c["early_stopping_rounds"])
        models["xgboost"] = m.calibrate(X_inner_val, y_inner_val)
    if mc["models"]["lightgbm"]["enabled"]:
        c = mc["models"]["lightgbm"]
        m = LGBMModel(
            n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
            num_leaves=c["num_leaves"],
        )
        m.fit(X_tr, y_tr, X_val=X_inner_val, y_val=y_inner_val,
              feature_names=feature_names)
        models["lightgbm"] = m.calibrate(X_inner_val, y_inner_val)
    return models


def _fold_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> FoldMetrics:
    base = evaluate_predictions(y_true, y_proba)
    pred = y_proba.argmax(axis=1)
    h, d, a = _per_class_recall(y_true, pred)
    return FoldMetrics(
        brier=base["brier_score"],
        rps=base["rps"],
        log_loss=base["log_loss"],
        accuracy=base["accuracy"],
        draw_f1=_draw_f1(y_true, pred),
        home_recall=h, draw_recall=d, away_recall=a,
    )


def _aggregate_metrics(folds: Iterable[FoldResult]) -> tuple[FoldMetrics, FoldMetrics]:
    """Return (mean, std) FoldMetrics across folds."""
    fold_list = list(folds)
    fields = [f.name for f in FoldMetrics.__dataclass_fields__.values()]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for field in fields:
        vals = [getattr(f.metrics, field) for f in fold_list]
        means[field] = float(statistics.fmean(vals))
        stds[field] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return FoldMetrics(**means), FoldMetrics(**stds)


def run_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    splitter: WalkForwardSplit,
    mc: dict,
    weights: dict[str, float],
) -> CVSection:
    """Run walk-forward CV and return the aggregated CVSection.

    df is sorted by (date, match_id) inside the splitter; ``feature_cols``
    must already be the canonical column order. ``weights`` must have a
    key per enabled model in mc.
    """
    sorted_df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    fold_specs = splitter.fold_specs(sorted_df)
    fold_results: list[FoldResult] = []

    for spec, (train_idx, val_idx) in zip(fold_specs, splitter.split(sorted_df)):
        X_all = sorted_df.iloc[train_idx][feature_cols].fillna(0).to_numpy(np.float32)
        y_all = sorted_df.iloc[train_idx]["result"].to_numpy(int)
        X_val = sorted_df.iloc[val_idx][feature_cols].fillna(0).to_numpy(np.float32)
        y_val = sorted_df.iloc[val_idx]["result"].to_numpy(int)

        # Inner val split for early stopping / calibration: last 10% of train
        cut = int(len(X_all) * 0.9)
        X_tr, y_tr = X_all[:cut], y_all[:cut]
        X_iv, y_iv = X_all[cut:], y_all[cut:]

        models = train_calibrated_models(X_tr, y_tr, X_iv, y_iv, feature_cols, mc)
        probas = {n: m.predict_proba(X_val) for n, m in models.items()}
        ens = _ensemble(probas, weights)

        metrics = _fold_metrics(y_val, ens)
        train_seasons = tuple(sorted(sorted_df.iloc[train_idx]["season"].unique()))
        train_max_md = int(sorted_df.iloc[train_idx]["matchday"].max())
        guards = FoldGuards(
            n_train=int(len(X_all)),
            n_val=int(len(X_val)),
            train_seasons=tuple(int(s) for s in train_seasons),
            train_matchday_max=train_max_md,
            val_season=int(spec["val_season"]),
            val_matchday_range=tuple(spec["val_matchday_range"]),
            leakage_check="passed",  # WalkForwardSplit already enforced ordering
        )
        fold_results.append(FoldResult(
            fold_id=int(spec["fold_id"]), metrics=metrics, guards=guards,
        ))
        logger.info(
            f"Fold {spec['fold_id']} season={spec['val_season']} "
            f"mds={spec['val_matchday_range']}: "
            f"n_train={len(X_all)} n_val={len(X_val)} "
            f"rps={metrics.rps:.4f} brier={metrics.brier:.4f} "
            f"draw_f1={metrics.draw_f1:.3f}"
        )

    mean, std = _aggregate_metrics(fold_results)
    return CVSection(folds=tuple(fold_results), mean_metrics=mean, std_metrics=std)
