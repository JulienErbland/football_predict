"""
xG (Expected Goals) feature builder.

Merges StatsBomb xG/PPDA data onto the main match DataFrame and computes
rolling averages per team. Returns None gracefully if StatsBomb data is absent.

Windows: [5, 10] — shorter windows are more responsive to recent form.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import settings


def _load_statsbomb_xg() -> pd.DataFrame | None:
    """Load all StatsBomb xG Parquet files and concatenate."""
    raw_dir = Path(settings()["paths"]["raw"]) / "statsbomb"
    files = list(raw_dir.glob("xg_*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def build_xg_features(
    matches: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame | None:
    """
    Compute rolling xG features per team.

    Returns None if StatsBomb data is not available (graceful degradation).
    Otherwise returns the matches DataFrame enriched with xG columns.

    Columns added:
        home_xg_w{w}_avg, away_xg_w{w}_avg  — rolling avg xG for/against
        home_xga_w{w}_avg, away_xga_w{w}_avg — rolling avg xG allowed
        home_xg_diff_w{w}, away_xg_diff_w{w} — xG - xGA (positive = good)
    """
    if windows is None:
        windows = [5, 10]

    xg_df = _load_statsbomb_xg()
    if xg_df is None or xg_df.empty:
        logger.warning("No StatsBomb xG data found — skipping xG features.")
        return None

    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Build per-team xG event view (home side)
    home_xg = xg_df[["match_id", "home_team", "home_xg", "away_xg",
                      "home_shots", "away_shots", "home_ppda", "away_ppda"]].copy()
    home_xg.columns = ["match_id", "team", "xg_for", "xg_against",
                        "shots_for", "shots_against", "ppda", "opp_ppda"]

    away_xg = xg_df[["match_id", "away_team", "away_xg", "home_xg",
                      "away_shots", "home_shots", "away_ppda", "home_ppda"]].copy()
    away_xg.columns = ["match_id", "team", "xg_for", "xg_against",
                        "shots_for", "shots_against", "ppda", "opp_ppda"]

    all_xg = pd.concat([home_xg, away_xg], ignore_index=True)
    # Merge dates from matches
    all_xg = all_xg.merge(
        df[["match_id", "date"]].drop_duplicates(), on="match_id", how="left"
    )
    all_xg = all_xg.sort_values("date")

    for w in windows:
        shifted = all_xg.groupby("team")[["xg_for", "xg_against", "ppda"]].shift(1)
        rolled = (
            shifted.groupby(all_xg["team"])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        all_xg[f"xg_for_w{w}"] = rolled["xg_for"].values
        all_xg[f"xg_against_w{w}"] = rolled["xg_against"].values
        all_xg[f"xg_diff_w{w}"] = (
            rolled["xg_for"].values - rolled["xg_against"].values
        )
        all_xg[f"ppda_w{w}"] = rolled["ppda"].values

    # Separate home/away back and merge onto matches
    home_rows = all_xg[all_xg["match_id"].isin(
        xg_df["match_id"]
    )].copy()
    # We stored home/away duplicates — identify by original side
    home_ids = set(xg_df["match_id"])
    home_part = all_xg[all_xg["match_id"].isin(home_ids) &
                        all_xg["team"].isin(xg_df["home_team"])]
    away_part = all_xg[all_xg["match_id"].isin(home_ids) &
                        all_xg["team"].isin(xg_df["away_team"])]

    xg_cols = [c for c in all_xg.columns if c.startswith("xg_") or c.startswith("ppda_")]
    home_feat = home_part.set_index("match_id")[xg_cols].add_prefix("home_")
    away_feat = away_part.set_index("match_id")[xg_cols].add_prefix("away_")

    df = df.merge(home_feat, on="match_id", how="left")
    df = df.merge(away_feat, on="match_id", how="left")
    logger.info(f"Built xG features ({len(xg_cols)*2} columns) for {len(df)} matches.")
    return df
