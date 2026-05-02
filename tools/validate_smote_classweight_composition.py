"""
T2.2 commit 2 — SMOTE+class_weight composition ablation harness.

Iterates a 6-cell grid (3 SMOTE sampling_strategies × 2 class_weight schemes),
runs walk-forward CV per cell on all 6 folds (or --folds subset), aggregates
per-fold (rps, brier, draw_f1, draw_recall, draw_precision) and applies the
margin filter decision rule to pick the winning cell.

Per design doc §2.2:
- Cells:
    1: smote='off',         weights=(1.0, 1.0, 1.0)
    2: smote='off',         weights=(1.0, 2.5, 1.2)
    3: smote='partial_70',  weights=(1.0, 1.0, 1.0)
    4: smote='partial_70',  weights=(1.0, 2.5, 1.2)
    5: smote='auto',        weights=(1.0, 1.0, 1.0)
    6: smote='auto',        weights=(1.0, 2.5, 1.2)
- Margin filter: keep cells where mean.rps <= 0.205 AND mean.brier <= 0.215.
- Pick max mean.draw_f1 among passing cells; tiebreak: lower std.draw_f1.
- If no cell passes margin: raise HarnessFailure (no auto-tune fallback).

CLI usage (from repo root):
    python -m tools.validate_smote_classweight_composition
    python -m tools.validate_smote_classweight_composition --folds 0,2,4
    python -m tools.validate_smote_classweight_composition --output path/to.json
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
from sklearn.metrics import f1_score, precision_score, recall_score


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from config.loader import settings, model_config  # noqa: E402
from evaluation.metrics import evaluate_predictions  # noqa: E402
from evaluation.splits import WalkForwardSplit  # noqa: E402
from features.build import FEATURE_SCHEMA_VERSION  # noqa: E402
from models.lgbm_model import LGBMModel  # noqa: E402
from models.xgboost_model import XGBoostModel  # noqa: E402
from training.draw_handling import class_sample_weights, resample  # noqa: E402


_META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}


_CELLS = [
    {"cell_id": 1, "smote_strategy": "off",         "class_weight": (1.0, 1.0, 1.0)},
    {"cell_id": 2, "smote_strategy": "off",         "class_weight": (1.0, 2.5, 1.2)},
    {"cell_id": 3, "smote_strategy": "partial_70", "class_weight": (1.0, 1.0, 1.0)},
    {"cell_id": 4, "smote_strategy": "partial_70", "class_weight": (1.0, 2.5, 1.2)},
    {"cell_id": 5, "smote_strategy": "auto",        "class_weight": (1.0, 1.0, 1.0)},
    {"cell_id": 6, "smote_strategy": "auto",        "class_weight": (1.0, 2.5, 1.2)},
]


class HarnessFailure(RuntimeError):
    """No cell cleared the margin filter — T2.2 cannot proceed."""


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _draw_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Argmax-based draw metrics (no theta_D applied — composition validation only)."""
    return {
        "draw_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "draw_precision": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "draw_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
    }


def _train_cell_and_predict(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_inner_val: np.ndarray, y_inner_val: np.ndarray,
    X_val: np.ndarray, feature_names: list[str],
    smote_strategy: str, weights_tuple: tuple[float, float, float],
    smote_k_neighbors: int, mc: dict,
) -> np.ndarray:
    """Apply SMOTE to (X_tr, y_tr), build sample_weight, fit calibrated XGB+LGBM, return ensemble proba on X_val."""
    X_tr_resampled, y_tr_resampled = resample(
        X_tr, y_tr,
        sampling_strategy=smote_strategy,
        k_neighbors=smote_k_neighbors,
    )
    sample_weight = class_sample_weights(
        y_tr_resampled,
        {"H": weights_tuple[0], "D": weights_tuple[1], "A": weights_tuple[2]},
    )

    probas: dict[str, np.ndarray] = {}

    xgb_cfg = mc["models"]["xgboost"]
    if xgb_cfg["enabled"]:
        m = XGBoostModel(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
        )
        m.fit(
            X_tr_resampled, y_tr_resampled,
            X_val=X_inner_val, y_val=y_inner_val,
            feature_names=feature_names,
            sample_weight=sample_weight,
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        )
        probas["xgboost"] = m.calibrate(X_inner_val, y_inner_val).predict_proba(X_val)

    lgbm_cfg = mc["models"]["lightgbm"]
    if lgbm_cfg["enabled"]:
        m = LGBMModel(
            n_estimators=lgbm_cfg["n_estimators"],
            learning_rate=lgbm_cfg["learning_rate"],
            num_leaves=lgbm_cfg["num_leaves"],
        )
        m.fit(
            X_tr_resampled, y_tr_resampled,
            X_val=X_inner_val, y_val=y_inner_val,
            feature_names=feature_names,
            sample_weight=sample_weight,
        )
        probas["lightgbm"] = m.calibrate(X_inner_val, y_inner_val).predict_proba(X_val)

    weights = {
        m_name: mc["models"][m_name]["weight"]
        for m_name in ("xgboost", "lightgbm")
        if mc["models"][m_name]["enabled"]
    }
    keys = list(probas)
    total = sum(weights[k] for k in keys)
    out = sum((weights[k] / total) * probas[k] for k in keys)
    out = np.clip(out, 0.0, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _aggregate(folds: list[dict]) -> tuple[dict, dict]:
    keys = ["rps", "brier", "draw_f1", "draw_precision", "draw_recall"]
    mean = {k: float(statistics.fmean([f[k] for f in folds])) for k in keys}
    std = {
        k: (float(statistics.pstdev([f[k] for f in folds])) if len(folds) > 1 else 0.0)
        for k in keys
    }
    return mean, std


def _select_winner(cells_out: list[dict]) -> dict:
    """Apply margin filter; return chosen cell.

    Margin: mean.rps <= 0.205 AND mean.brier <= 0.215 (gates with 0.005 margin).
    Pick max mean.draw_f1 (tiebreak: lower std.draw_f1).
    """
    passing = [
        c for c in cells_out
        if c["mean"]["rps"] <= 0.205 and c["mean"]["brier"] <= 0.215
    ]
    if not passing:
        raise HarnessFailure(
            "No cell within margin (rps <= 0.205, brier <= 0.215). "
            "T2.2's three-pronged approach is insufficient. "
            "Retrospective should address fourth-mechanism candidates "
            "(focal loss, weighted Brier, ordinal regression). "
            "Meta-spec re-open required."
        )
    passing.sort(key=lambda c: (-c["mean"]["draw_f1"], c["std"]["draw_f1"]))
    return passing[0]


def run(
    features_path: Path | None = None,
    output_path: Path | None = None,
    folds_subset: list[int] | None = None,
    smote_k_neighbors: int = 5,
) -> dict:
    cfg = settings()
    mc = model_config()

    if features_path is None:
        features_path = Path(cfg["paths"]["features"])
    if output_path is None:
        output_path = Path(cfg["paths"]["output"]) / "smote_classweight_ablation.json"

    if not features_path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {features_path}. Run features.build first."
        )

    df = pd.read_parquet(features_path).sort_values(["date", "match_id"]).reset_index(drop=True)
    feat_cols = _feature_cols(df)

    splitter = WalkForwardSplit()
    fold_specs = splitter.fold_specs(df)
    fold_pairs = list(splitter.split(df))
    if folds_subset is not None:
        fold_specs = [fold_specs[i] for i in folds_subset]
        fold_pairs = [fold_pairs[i] for i in folds_subset]

    logger.info(
        f"Loaded {len(df)} rows x {len(feat_cols)} features. "
        f"Running {len(_CELLS)} cells x {len(fold_specs)} folds = "
        f"{len(_CELLS) * len(fold_specs)} fold-runs."
    )

    started_at = datetime.now(timezone.utc).isoformat()
    cells_out: list[dict] = []

    for cell in _CELLS:
        logger.info(
            f"=== Cell {cell['cell_id']}: smote={cell['smote_strategy']!r} "
            f"class_weight={cell['class_weight']} ==="
        )
        per_fold: list[dict] = []
        for spec, (train_idx, val_idx) in zip(fold_specs, fold_pairs):
            t0 = time.time()
            X_all = df.iloc[train_idx][feat_cols].fillna(0).to_numpy(np.float32)
            y_all = df.iloc[train_idx]["result"].to_numpy(int)
            X_val = df.iloc[val_idx][feat_cols].fillna(0).to_numpy(np.float32)
            y_val = df.iloc[val_idx]["result"].to_numpy(int)
            cut = int(len(X_all) * 0.9)
            X_tr, y_tr = X_all[:cut], y_all[:cut]
            X_iv, y_iv = X_all[cut:], y_all[cut:]

            ens_proba = _train_cell_and_predict(
                X_tr, y_tr, X_iv, y_iv, X_val, feat_cols,
                smote_strategy=cell["smote_strategy"],
                weights_tuple=cell["class_weight"],
                smote_k_neighbors=smote_k_neighbors,
                mc=mc,
            )
            base = evaluate_predictions(y_val, ens_proba)
            pred = ens_proba.argmax(axis=1)
            draw_m = _draw_metrics(y_val, pred)

            fold_record = {
                "fold_id": int(spec["fold_id"]),
                "rps": base["rps"],
                "brier": base["brier_score"],
                **draw_m,
            }
            per_fold.append(fold_record)
            logger.info(
                f"  fold={spec['fold_id']} season={spec['val_season']} "
                f"rps={base['rps']:.4f} brier={base['brier_score']:.4f} "
                f"draw_f1={draw_m['draw_f1']:.3f} ({time.time()-t0:.1f}s)"
            )

        mean, std = _aggregate(per_fold)
        cells_out.append({
            "cell_id": cell["cell_id"],
            "smote_strategy": cell["smote_strategy"],
            "class_weight": list(cell["class_weight"]),
            "fold_results": per_fold,
            "mean": mean,
            "std": std,
        })
        logger.info(
            f"  AGGREGATE: mean.rps={mean['rps']:.4f} mean.brier={mean['brier']:.4f} "
            f"mean.draw_f1={mean['draw_f1']:.3f} std.draw_f1={std['draw_f1']:.3f}"
        )

    # Build the report skeleton with cells_out captured BEFORE applying the
    # margin filter, so a HarnessFailure preserves measurement evidence on
    # disk instead of being lost to the traceback. Winner is filled in
    # post-selection; on failure it stays null and failure_reason records
    # why no cell was chosen.
    report: dict = {
        "schema_version": "smote_cw_ablation.v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at": started_at,
        "smote_k_neighbors": smote_k_neighbors,
        "folds_used": [int(s["fold_id"]) for s in fold_specs],
        "cells": cells_out,
        "winner": None,
        "failure_reason": None,
    }
    try:
        winner = _select_winner(cells_out)
    except HarnessFailure as exc:
        report["failure_reason"] = str(exc)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        logger.error(f"Wrote {output_path} (HarnessFailure — winner=null)")
        raise

    logger.info(
        f"WINNING CELL: id={winner['cell_id']} smote={winner['smote_strategy']!r} "
        f"class_weight={winner['class_weight']} "
        f"mean.draw_f1={winner['mean']['draw_f1']:.3f}"
    )
    report["winner"] = {
        "cell_id": winner["cell_id"],
        "smote_strategy": winner["smote_strategy"],
        "class_weight": winner["class_weight"],
        "rationale": (
            f"Highest mean.draw_f1 ({winner['mean']['draw_f1']:.3f}) "
            "among cells passing margin filter (rps <= 0.205, brier <= 0.215). "
            f"std.draw_f1 = {winner['std']['draw_f1']:.3f}."
        ),
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
                        help="Output JSON path (default: data/output/smote_classweight_ablation.json).")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold indices to run (e.g. '0,2,4'). Default: all.")
    parser.add_argument("--smote-k-neighbors", type=int, default=5,
                        help="SMOTE k_neighbors (default: 5).")
    args = parser.parse_args()

    folds_subset = None
    if args.folds:
        folds_subset = [int(x) for x in args.folds.split(",")]

    run(
        features_path=args.features,
        output_path=args.output,
        folds_subset=folds_subset,
        smote_k_neighbors=args.smote_k_neighbors,
    )


if __name__ == "__main__":
    main()
