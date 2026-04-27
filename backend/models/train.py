"""
Training orchestrator for the football prediction ensemble.

Workflow:
    1. Load features.parquet
    2. Time-based train/test split by season (never random)
    3. Train all enabled models  [PLACEHOLDERS — see note below]
    4. Calibrate each with isotonic regression
    5. Build weighted average ensemble
    6. Save models to data/models/

HOW TO TRAIN THE MODEL (run these commands):
    cd backend
    # Option A — free, no API key needed (football-data.co.uk CSVs):
    python -m ingestion.football_data_csv --leagues PL,PD,BL1,SA,FL1 --seasons 2021,2022,2023,2024
    # Option B — football-data.org API (requires paid tier for historical seasons):
    # python -m ingestion.football_data --leagues PL,PD,BL1,SA,FL1 --seasons 2021,2022,2023,2024
    python -m features.build
    python -m models.train

Expected runtime: ~30-60 minutes on CPU. 8 GB RAM minimum.

NOTE: per project instructions, model.fit() calls are placeholders in this file.
      Uncomment the fit() blocks to actually train. See PROJECT.md for full guide.

Time-based split is critical: random splits would leak future form ratings, Elo,
and league positions into the training set, inflating test performance.
"""

from __future__ import annotations

import argparse
import pickle  # Used for feature_cols only — ML objects serialised by their own .save()
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import settings, model_config
from models.poisson_model import PoissonModel
from models.xgboost_model import XGBoostModel
from models.lgbm_model import LGBMModel
from models.ensemble import EnsembleModel


# Columns that are metadata, not features
_META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {path}. "
            "Run `python -m features.build` first."
        )
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _time_split(df: pd.DataFrame, test_seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by season — all test_seasons go into test set, everything prior into train.
    This preserves temporal ordering and avoids any form of future leakage.
    """
    test_mask = df["season"].isin(test_seasons)
    train = df[~test_mask & df["result"].notna()]
    test = df[test_mask & df["result"].notna()]
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


def _prepare_X_y(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["result"].values.astype(int)
    return X, y


def train(force_retrain: bool = False) -> EnsembleModel:
    """Run the training pipeline (fit() calls are placeholders)."""
    cfg = settings()
    mc = model_config()

    features_path = Path(cfg["paths"]["features"])
    models_dir = Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    df = _load_features(features_path)
    feature_cols = _get_feature_cols(df)
    logger.info(f"Feature columns: {len(feature_cols)}")

    test_seasons = mc["evaluation"]["test_seasons"]
    train_df, test_df = _time_split(df, test_seasons)

    X_train, y_train = _prepare_X_y(train_df, feature_cols)
    X_test, y_test = _prepare_X_y(test_df, feature_cols)

    # Validation split from end of training set for early stopping (last 10%)
    val_idx = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_idx:], y_train[val_idx:]
    X_tr, y_tr = X_train[:val_idx], y_train[:val_idx]

    trained_models: dict = {}
    weights: dict[str, float] = {}

    # ── Poisson Model ──────────────────────────────────────────────────────
    if mc["models"]["poisson"]["enabled"]:
        logger.info("Training Poisson model...")
        poisson = PoissonModel()

        # ── TRAINING PLACEHOLDER ──────────────────────────────────────────
        # Uncomment to actually train:
        # poisson.fit(
        #     X_train, y_train,
        #     home_teams=train_df["home_team"].tolist(),
        #     away_teams=train_df["away_team"].tolist(),
        #     home_goals=train_df["home_goals"].tolist(),
        #     away_goals=train_df["away_goals"].tolist(),
        # )
        # calibrated_poisson = poisson.calibrate(X_val, y_val)
        # calibrated_poisson.save(models_dir / "poisson.pkl")
        logger.info("Poisson training skipped (placeholder).")
        # ─────────────────────────────────────────────────────────────────

        trained_models["poisson"] = poisson
        weights["poisson"] = mc["models"]["poisson"]["weight"]

    # ── XGBoost ───────────────────────────────────────────────────────────
    if mc["models"]["xgboost"]["enabled"]:
        logger.info("Training XGBoost model...")
        xgb_model = XGBoostModel(
            n_estimators=mc["models"]["xgboost"]["n_estimators"],
            learning_rate=mc["models"]["xgboost"]["learning_rate"],
            max_depth=mc["models"]["xgboost"]["max_depth"],
            subsample=mc["models"]["xgboost"]["subsample"],
            colsample_bytree=mc["models"]["xgboost"]["colsample_bytree"],
        )

        xgb_model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val,
                      feature_names=feature_cols,
                      early_stopping_rounds=mc["models"]["xgboost"]["early_stopping_rounds"])
        calibrated_xgb = xgb_model.calibrate(X_val, y_val)
        calibrated_xgb.save(models_dir / "xgboost.pkl")

        trained_models["xgboost"] = calibrated_xgb
        weights["xgboost"] = mc["models"]["xgboost"]["weight"]

    # ── LightGBM ──────────────────────────────────────────────────────────
    if mc["models"]["lightgbm"]["enabled"]:
        logger.info("Training LightGBM model...")
        lgbm_model = LGBMModel(
            n_estimators=mc["models"]["lightgbm"]["n_estimators"],
            learning_rate=mc["models"]["lightgbm"]["learning_rate"],
            num_leaves=mc["models"]["lightgbm"]["num_leaves"],
        )

        lgbm_model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val,
                       feature_names=feature_cols)
        calibrated_lgbm = lgbm_model.calibrate(X_val, y_val)
        calibrated_lgbm.save(models_dir / "lightgbm.pkl")

        trained_models["lightgbm"] = calibrated_lgbm
        weights["lightgbm"] = mc["models"]["lightgbm"]["weight"]

    # ── Neural Net (optional) ──────────────────────────────────────────────
    if mc["models"]["neural_net"]["enabled"] and len(X_train) > 2000:
        from models.neural_net import NeuralNetModel
        nn_model = NeuralNetModel(
            hidden_sizes=mc["models"]["neural_net"]["hidden_sizes"],
            dropout=mc["models"]["neural_net"]["dropout"],
            epochs=mc["models"]["neural_net"]["epochs"],
        )
        # ── TRAINING PLACEHOLDER ─────────────────────────────────────────
        # nn_model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        # calibrated_nn = nn_model.calibrate(X_val, y_val)
        # calibrated_nn.save(models_dir / "neural_net.pkl")
        # ─────────────────────────────────────────────────────────────────
        trained_models["neural_net"] = nn_model
        weights["neural_net"] = mc["models"]["neural_net"]["weight"]

    # ── Ensemble ──────────────────────────────────────────────────────────
    ensemble = EnsembleModel(trained_models, weights)
    ensemble_path = models_dir / "ensemble.pkl"
    ensemble.save(ensemble_path)
    logger.info(f"Saved ensemble → {ensemble_path}")

    # ── Save feature column order ──────────────────────────────────────────
    # CRITICAL: inference must use features in the exact same order as training
    feature_cols_path = models_dir / "feature_cols.pkl"
    with open(feature_cols_path, "wb") as f:
        pickle.dump(feature_cols, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved feature column order → {feature_cols_path}")

    logger.info("Training pipeline complete.")
    return ensemble


def main():
    parser = argparse.ArgumentParser(description="Train football prediction models")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if models already exist")
    args = parser.parse_args()
    train(force_retrain=args.force)


if __name__ == "__main__":
    main()
