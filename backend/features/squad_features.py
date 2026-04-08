"""
Squad feature builder.

Merges Transfermarkt squad market values onto matches by team name.
Returns None gracefully if Transfermarkt data is absent.

Features:
    home_squad_value_eur_m, away_squad_value_eur_m — total squad value in €M
    squad_value_ratio — home / away (>1 means home team is richer)
    home_avg_age, away_avg_age — squad average age
    age_difference — home - away avg age
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from config.loader import settings


def _load_squad_values() -> pd.DataFrame | None:
    """Load all Transfermarkt squad value Parquet files."""
    raw_dir = Path(settings()["paths"]["raw"]) / "transfermarkt"
    files = list(raw_dir.glob("*_squad_values.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _load_injury_counts() -> pd.DataFrame | None:
    """
    Aggregate injury files into a per-team injury count.

    Returns DataFrame with columns: team_name, injury_count
    """
    raw_dir = Path(settings()["paths"]["raw"]) / "transfermarkt"
    files = list(raw_dir.glob("injuries_*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby("team_name").size().reset_index(name="injury_count")


def build_squad_features(matches: pd.DataFrame) -> pd.DataFrame | None:
    """
    Merge squad value and injury features onto matches.

    Returns None if no Transfermarkt data is available.
    """
    squad_df = _load_squad_values()
    if squad_df is None or squad_df.empty:
        logger.warning("No Transfermarkt squad data — skipping squad features.")
        return None

    injury_df = _load_injury_counts()

    # Deduplicate: if multiple seasons scraped, take most recent per team
    squad_df = (
        squad_df.sort_values("team_name")
        .drop_duplicates(subset=["team_name"], keep="last")
    )

    # Build a normalised name-to-value lookup
    value_map = squad_df.set_index("team_name")["total_value_eur_m"].to_dict()
    age_map = squad_df.set_index("team_name")["avg_age"].to_dict()
    injury_map: dict = {}
    if injury_df is not None:
        injury_map = injury_df.set_index("team_name")["injury_count"].to_dict()

    df = matches.copy()
    df["home_squad_value_eur_m"] = df["home_team"].map(value_map)
    df["away_squad_value_eur_m"] = df["away_team"].map(value_map)
    df["squad_value_ratio"] = df["home_squad_value_eur_m"] / df["away_squad_value_eur_m"]
    df["home_avg_age"] = df["home_team"].map(age_map)
    df["away_avg_age"] = df["away_team"].map(age_map)
    df["age_difference"] = df["home_avg_age"] - df["away_avg_age"]
    df["home_injury_count"] = df["home_team"].map(injury_map).fillna(0).astype(int)
    df["away_injury_count"] = df["away_team"].map(injury_map).fillna(0).astype(int)

    coverage = df["home_squad_value_eur_m"].notna().mean()
    logger.info(f"Squad features built. Coverage: {coverage:.1%} of matches.")
    return df
