"""
Master feature orchestrator.

Loads all_matches.parquet, calls each enabled feature builder, merges
results on match_id, and saves the final flat feature table to
data/features/features.parquet.

Each optional builder (xG, squad, tactical) returns None on failure —
build.py silently skips that group and logs a warning. The pipeline
can always run using only football-data.org features.

CLI usage:
    python -m features.build
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from config.loader import settings, feature_config
from features.elo import compute_elo
from features.form import build_form_features, build_h2h_features
from features.xg_features import build_xg_features
from features.squad_features import build_squad_features
from features.tactical import build_tactical_features
from features.context import build_context_features


def build_features(matches_path: str | Path | None = None) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Args:
        matches_path: Optional path to all_matches.parquet. Defaults to
            the path in settings.yaml.

    Returns:
        Feature DataFrame saved to data/features/features.parquet.
    """
    cfg = settings()
    fc = feature_config()

    if matches_path is None:
        matches_path = Path(cfg["paths"]["processed"]) / "all_matches.parquet"

    logger.info(f"Loading matches from {matches_path}")
    if not Path(matches_path).exists():
        raise FileNotFoundError(
            f"all_matches.parquet not found at {matches_path}. "
            "Run `python -m ingestion.football_data` first."
        )
    df = pd.read_parquet(matches_path)
    logger.info(f"Loaded {len(df)} matches.")

    # ── Elo ────────────────────────────────────────────────────────────────
    if fc.get("elo", {}).get("enabled", True):
        logger.info("Building Elo features...")
        df = compute_elo(df)

    # ── Form (rolling stats) ───────────────────────────────────────────────
    if fc.get("form", {}).get("enabled", True):
        windows = fc.get("form", {}).get("windows", [3, 5, 10])
        logger.info(f"Building form features (windows={windows})...")
        df = build_form_features(df, windows=windows)

    # ── Head-to-head ───────────────────────────────────────────────────────
    if fc.get("head_to_head", {}).get("enabled", True):
        window_years = fc.get("head_to_head", {}).get("window_years", 5)
        logger.info(f"Building H2H features (lookback={window_years}y)...")
        df = build_h2h_features(df, window_years=window_years)

    # ── xG features (optional — StatsBomb) ────────────────────────────────
    if fc.get("xg", {}).get("enabled", True):
        xg_windows = fc.get("xg", {}).get("windows", [5, 10])
        logger.info(f"Building xG features (windows={xg_windows})...")
        xg_result = build_xg_features(df, windows=xg_windows)
        if xg_result is not None:
            df = xg_result
        else:
            logger.warning("xG features skipped — StatsBomb data unavailable.")

    # ── Squad features (optional — Transfermarkt) ─────────────────────────
    if fc.get("squad", {}).get("enabled", True):
        logger.info("Building squad features...")
        squad_result = build_squad_features(df)
        if squad_result is not None:
            df = squad_result
        else:
            logger.warning("Squad features skipped — Transfermarkt data unavailable.")

    # ── Tactical features (optional — StatsBomb lineups) ──────────────────
    if fc.get("tactics", {}).get("enabled", True):
        logger.info("Building tactical features...")
        tactical_result = build_tactical_features(df)
        if tactical_result is not None:
            df = tactical_result
        else:
            logger.warning("Tactical features skipped — lineup data unavailable.")

    # ── Context features (always available) ───────────────────────────────
    if fc.get("context", {}).get("enabled", True):
        logger.info("Building context features...")
        df = build_context_features(df)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path(cfg["paths"]["features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    feature_cols = [c for c in df.columns
                    if c not in ("match_id", "league", "season", "date",
                                  "home_team", "away_team", "home_team_id", "away_team_id",
                                  "matchday", "referee", "result",
                                  "home_goals", "away_goals")]
    logger.info(
        f"Feature table saved to {out_path}. "
        f"{len(df)} rows × {len(feature_cols)} feature columns."
    )
    return df


def main():
    build_features()


if __name__ == "__main__":
    main()
