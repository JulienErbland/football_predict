"""
T2.1 commit 2 — empirical CV parametrization harness.

For each candidate (n_splits_per_season, val_window_matchdays) parametrization,
runs walk-forward CV across the CV pool (2021/2022/2023, with 2024 reserved as
the future locked holdout) using XGBoost + LightGBM (per `model_config.yaml`),
builds a weighted-average ensemble per fold, and captures per-fold metrics
plus class distribution. Aggregates per-config mean/std and writes
`backend/data/output/cv_parametrization_validation.json`.

This is a one-shot rationale tool — it lives at repo root (not under
`backend/`) because it duplicates a minimal walk-forward iterator that
becomes redundant once `backend/evaluation/splits.WalkForwardSplit`
(commit 3) lands. Re-run only if the candidate set or CV pool changes.

CLI usage (from repo root):
    python -m tools.validate_cv_parametrization
    python -m tools.validate_cv_parametrization --quick     # 1 fold/config smoke
    python -m tools.validate_cv_parametrization --output path/to.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from config.loader import settings, model_config  # noqa: E402
from evaluation.metrics import evaluate_predictions  # noqa: E402
from models.xgboost_model import XGBoostModel  # noqa: E402
from models.lgbm_model import LGBMModel  # noqa: E402


_CV_POOL_SEASONS = (2021, 2022, 2023)
_HOLDOUT_SEASON = 2024
_META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}

_CANDIDATES = [
    {"n_splits": 3, "val_window": 6, "label": "(3, 6)"},
    {"n_splits": 4, "val_window": 5, "label": "(4, 5)"},
    {"n_splits": 2, "val_window": 9, "label": "(2, 9)"},
]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _val_window_starts(max_md: int, n_splits: int, vw: int) -> list[int]:
    """Evenly spaced val-window starts within a season; warmup = vw matchdays."""
    earliest = vw + 1
    latest = max_md - vw + 1
    if latest < earliest:
        return []
    if n_splits == 1:
        return [latest]
    step = (latest - earliest) / (n_splits - 1)
    return sorted({round(earliest + i * step) for i in range(n_splits)})


def _fold_definitions(df: pd.DataFrame, n_splits: int, vw: int) -> list[dict]:
    """
    Build fold definitions for the (n_splits, vw) config across the CV pool.

    Each fold dict has: season, val_md_range (start, end), val_start_date,
    train_idx (np.ndarray of df row positions), val_idx (np.ndarray).
    Train = all CV-pool rows dated strictly before the val window's earliest
    date. Val = rows in `season` whose matchday is in [val_start, val_end].
    """
    pool = df[df["season"].isin(_CV_POOL_SEASONS)].sort_values("date").reset_index(drop=True)
    pool_dates = pd.to_datetime(pool["date"])

    folds: list[dict] = []
    for season in _CV_POOL_SEASONS:
        season_rows = pool[pool["season"] == season]
        if season_rows.empty:
            continue
        # Use the per-(league,season) max-matchday min — windows must fit every league
        max_md = int(season_rows.groupby("league")["matchday"].max().min())
        starts = _val_window_starts(max_md, n_splits, vw)
        for s_md in starts:
            e_md = min(s_md + vw - 1, max_md)
            in_window = (
                (pool["season"] == season)
                & (pool["matchday"] >= s_md)
                & (pool["matchday"] <= e_md)
            )
            if not in_window.any():
                continue
            val_start_date = pool_dates[in_window].min()
            train_mask = pool_dates < val_start_date
            val_idx = pool.index[in_window].to_numpy()
            train_idx = pool.index[train_mask].to_numpy()
            folds.append({
                "season": season,
                "val_md_range": (s_md, e_md),
                "val_start_date": val_start_date.isoformat(),
                "train_idx": train_idx,
                "val_idx": val_idx,
            })
    return folds


def _draw_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """F1 of the draw class against argmax predictions."""
    pred = y_proba.argmax(axis=1)
    return float(f1_score(y_true, pred, labels=[1], average="macro", zero_division=0))


def _train_and_predict(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: list[str],
    mc: dict,
) -> dict[str, np.ndarray]:
    """Train enabled models, return {model_name: y_proba} on X_val."""
    out: dict[str, np.ndarray] = {}

    # Inner val split for early stopping / calibration: last 10% of train
    cut = int(len(X_tr) * 0.9)
    X_inner_tr, y_inner_tr = X_tr[:cut], y_tr[:cut]
    X_inner_val, y_inner_val = X_tr[cut:], y_tr[cut:]

    if mc["models"]["xgboost"]["enabled"]:
        xgb_cfg = mc["models"]["xgboost"]
        xgb_m = XGBoostModel(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
        )
        xgb_m.fit(
            X_inner_tr, y_inner_tr,
            X_val=X_inner_val, y_val=y_inner_val,
            feature_names=feature_names,
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        )
        out["xgboost"] = xgb_m.calibrate(X_inner_val, y_inner_val).predict_proba(X_val)

    if mc["models"]["lightgbm"]["enabled"]:
        lgbm_cfg = mc["models"]["lightgbm"]
        lgbm_m = LGBMModel(
            n_estimators=lgbm_cfg["n_estimators"],
            learning_rate=lgbm_cfg["learning_rate"],
            num_leaves=lgbm_cfg["num_leaves"],
        )
        lgbm_m.fit(X_inner_tr, y_inner_tr,
                   X_val=X_inner_val, y_val=y_inner_val,
                   feature_names=feature_names)
        out["lightgbm"] = lgbm_m.calibrate(X_inner_val, y_inner_val).predict_proba(X_val)

    return out


def _ensemble(probas: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of calibrated probas, weights renormalised to enabled set.

    Clips into [0, 1] and renormalises rows to sum to exactly 1.0 (fp32
    drift after calibration + averaging can leave row sums at 1+1e-7,
    which sklearn.log_loss rejects).
    """
    keys = list(probas)
    total = sum(weights[k] for k in keys)
    out = sum((weights[k] / total) * probas[k] for k in keys)
    out = np.clip(out, 0.0, 1.0)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _class_distribution(y: np.ndarray) -> dict[str, float]:
    n = len(y)
    if n == 0:
        return {"home": 0.0, "draw": 0.0, "away": 0.0}
    return {
        "home": float((y == 0).mean()),
        "draw": float((y == 1).mean()),
        "away": float((y == 2).mean()),
    }


def _aggregate(folds_metrics: list[dict]) -> dict:
    """Return per-key mean/std across folds."""
    keys = ["brier_score", "log_loss", "rps", "accuracy", "draw_f1",
            "n_samples", "draw_rate"]
    out = {}
    for k in keys:
        vals = [f[k] for f in folds_metrics if k in f]
        if not vals:
            continue
        out[k] = {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return out


def run(
    features_path: Path | None = None,
    output_path: Path | None = None,
    quick: bool = False,
) -> dict:
    cfg = settings()
    mc = model_config()

    if features_path is None:
        features_path = Path(cfg["paths"]["features"])
    if output_path is None:
        output_path = Path(cfg["paths"]["output"]) / "cv_parametrization_validation.json"

    if not features_path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {features_path}. Run features.build first."
        )

    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)
    feat_cols = _feature_cols(df)
    logger.info(
        f"Loaded {len(df)} rows × {len(feat_cols)} features from {features_path}; "
        f"CV pool seasons {_CV_POOL_SEASONS}, holdout {_HOLDOUT_SEASON}."
    )
    pool = df[df["season"].isin(_CV_POOL_SEASONS)].sort_values("date").reset_index(drop=True)

    weights = {
        m: mc["models"][m]["weight"]
        for m in ("xgboost", "lightgbm")
        if mc["models"][m]["enabled"]
    }
    if not weights:
        raise RuntimeError("No models enabled in model_config.yaml.")

    started_at = datetime.now(timezone.utc).isoformat()
    candidates_out: list[dict] = []

    for cand in _CANDIDATES:
        n_splits, vw = cand["n_splits"], cand["val_window"]
        logger.info(f"=== Candidate {cand['label']}  n_splits={n_splits}  vw={vw} ===")
        folds = _fold_definitions(pool, n_splits, vw)
        if quick:
            folds = folds[:1]
        logger.info(f"  {len(folds)} folds.")

        per_fold: list[dict] = []
        for i, fold in enumerate(folds, 1):
            t0 = time.time()
            train_idx = fold["train_idx"]
            val_idx = fold["val_idx"]
            X_tr = pool.iloc[train_idx][feat_cols].fillna(0).to_numpy(np.float32)
            y_tr = pool.iloc[train_idx]["result"].to_numpy(int)
            X_val = pool.iloc[val_idx][feat_cols].fillna(0).to_numpy(np.float32)
            y_val = pool.iloc[val_idx]["result"].to_numpy(int)

            if len(X_val) < 100:
                logger.warning(
                    f"  Fold {i}: skipping — n_val={len(X_val)} below sanity floor (100)."
                )
                continue

            probas = _train_and_predict(X_tr, y_tr, X_val, y_val, feat_cols, mc)
            ens = _ensemble(probas, weights)
            base = evaluate_predictions(y_val, ens)
            base["draw_f1"] = _draw_f1(y_val, ens)
            cd = _class_distribution(y_val)
            base["draw_rate"] = cd["draw"]

            fold_record = {
                "season": fold["season"],
                "val_md_range": list(fold["val_md_range"]),
                "val_start_date": fold["val_start_date"],
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_val)),
                "class_distribution": cd,
                **base,
            }
            per_fold.append(fold_record)
            logger.info(
                f"  Fold {i}/{len(folds)} season={fold['season']} "
                f"mds={fold['val_md_range']}: "
                f"n_train={len(X_tr)} n_val={len(X_val)} "
                f"rps={base['rps']:.4f} brier={base['brier_score']:.4f} "
                f"draw_f1={base['draw_f1']:.3f} "
                f"draw_rate={cd['draw']:.2%} ({time.time()-t0:.1f}s)"
            )

        candidates_out.append({
            "label": cand["label"],
            "n_splits_per_season": n_splits,
            "val_window_matchdays": vw,
            "n_folds": len(per_fold),
            "folds": per_fold,
            "aggregate": _aggregate(per_fold),
        })

    report = {
        "schema_version": "cv_parametrization_validation.v1",
        "generated_at": started_at,
        "cv_pool_seasons": list(_CV_POOL_SEASONS),
        "holdout_season": _HOLDOUT_SEASON,
        "weights": weights,
        "candidates": candidates_out,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=None,
                        help="Path to features.parquet (default: from settings).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: data/output/cv_parametrization_validation.json).")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke mode: 1 fold per candidate.")
    args = parser.parse_args()

    run(features_path=args.features, output_path=args.output, quick=args.quick)


if __name__ == "__main__":
    main()
