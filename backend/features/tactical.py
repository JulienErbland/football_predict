"""
Tactical feature builder.

Derives features from formation strings and lineup compositions.
Returns None gracefully if lineup/formation data is not available.

Formation encoding:
    We assign integer IDs to common formations. The pair (home_formation_id,
    away_formation_id) is hashed as home_id * 100 + away_id — compact enough
    to use as a lookup key for historical win rates per matchup type.

Player archetype scoring:
    Pace advantage — proportion of wingers/fullbacks (fast-running positions)
    Aerial advantage — proportion of centre-backs/target-forwards (tall positions)
    Technical advantage — proportion of midfielders/10s (skill-heavy positions)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from config.loader import settings


_FORMATION_MAP = {
    "4-4-2": 1, "4-3-3": 2, "4-2-3-1": 3, "4-5-1": 4,
    "3-5-2": 5, "3-4-3": 6, "5-3-2": 7, "5-4-1": 8,
    "4-1-4-1": 9, "4-4-1-1": 10, "3-4-2-1": 11, "4-2-2-2": 12,
}

_PACE_POSITIONS = {"Left Winger", "Right Winger", "Left Back", "Right Back",
                    "Left Midfield", "Right Midfield"}
_AERIAL_POSITIONS = {"Centre Back", "Left Centre Back", "Right Centre Back",
                      "Centre Forward", "Striker"}
_TECHNICAL_POSITIONS = {"Central Midfield", "Attacking Midfield", "Defensive Midfield",
                         "Right Defensive Midfield", "Left Defensive Midfield"}


def _load_statsbomb_formations() -> pd.DataFrame | None:
    raw_dir = Path(settings()["paths"]["raw"]) / "statsbomb"
    files = list(raw_dir.glob("formations_*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _formation_id(formation_str: str | None) -> int:
    if not formation_str:
        return 0
    return _FORMATION_MAP.get(str(formation_str).strip(), 0)


def _archetype_scores(lineup_df: pd.DataFrame, team: str) -> tuple[float, float, float]:
    """Return (pace, aerial, technical) advantage scores for a team's lineup."""
    players = lineup_df[lineup_df["team"] == team]
    if players.empty:
        return 0.0, 0.0, 0.0
    positions = set(players["player_position"].dropna())
    n = len(players)
    pace = len(positions & _PACE_POSITIONS) / max(n, 1)
    aerial = len(positions & _AERIAL_POSITIONS) / max(n, 1)
    technical = len(positions & _TECHNICAL_POSITIONS) / max(n, 1)
    return pace, aerial, technical


def build_tactical_features(matches: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build formation and player archetype features.

    Returns None if StatsBomb lineup data is not available.
    """
    formations_df = _load_statsbomb_formations()
    if formations_df is None or formations_df.empty:
        logger.warning("No StatsBomb lineup data — skipping tactical features.")
        return None

    df = matches.copy()

    # ── Per-match formation lookup ─────────────────────────────────────────
    # StatsBomb match_ids may differ from football-data.org match_ids.
    # We join on (home_team, away_team, date) if possible; fallback: no-join.
    home_form = (
        formations_df[formations_df["team"].isin(df["home_team"].unique())]
        .groupby(["match_id", "team"])["formation"]
        .first()
        .reset_index()
    )
    # Build a match_id → formation dict for home and away sides
    match_formations: dict[int, dict] = {}
    for _, row in home_form.iterrows():
        mid = row["match_id"]
        if mid not in match_formations:
            match_formations[mid] = {}
        match_formations[mid][row["team"]] = row["formation"]

    # ── Build formation pair history for win-rate lookup ───────────────────
    formation_records = []
    for mid, fm in match_formations.items():
        match_row = df[df["match_id"] == mid]
        if match_row.empty:
            continue
        match_row = match_row.iloc[0]
        home_f = fm.get(match_row["home_team"])
        away_f = fm.get(match_row["away_team"])
        h_id = _formation_id(home_f)
        a_id = _formation_id(away_f)
        formation_records.append({
            "match_id": mid,
            "home_formation_id": h_id,
            "away_formation_id": a_id,
            "formation_pair_hash": h_id * 100 + a_id,
            "result": match_row.get("result"),
        })

    form_df = pd.DataFrame(formation_records)
    if form_df.empty:
        return None

    # ── Historical win rate per formation pair (using all past matches) ────
    pair_wins = (
        form_df[form_df["result"].notna()]
        .groupby("formation_pair_hash")["result"]
        .agg(
            pair_home_win_rate=lambda x: (x == 0).mean(),
            pair_draw_rate=lambda x: (x == 1).mean(),
            pair_away_win_rate=lambda x: (x == 2).mean(),
            pair_sample_size="count",
        )
        .reset_index()
    )

    # ── Archetype scores per match ─────────────────────────────────────────
    archetype_rows = []
    for mid, row in form_df.iterrows():
        lineup = formations_df[formations_df["match_id"] == row["match_id"]]
        match_ref = df[df["match_id"] == row["match_id"]]
        if match_ref.empty:
            continue
        home = match_ref.iloc[0]["home_team"]
        away = match_ref.iloc[0]["away_team"]
        h_pace, h_aerial, h_tech = _archetype_scores(lineup, home)
        a_pace, a_aerial, a_tech = _archetype_scores(lineup, away)
        archetype_rows.append({
            "match_id": row["match_id"],
            "pace_advantage": h_pace - a_pace,
            "aerial_advantage": h_aerial - a_aerial,
            "technical_advantage": h_tech - a_tech,
        })
    archetype_df = pd.DataFrame(archetype_rows)

    # ── Merge everything onto main DataFrame ──────────────────────────────
    df = df.merge(form_df[["match_id", "home_formation_id", "away_formation_id",
                             "formation_pair_hash"]],
                   on="match_id", how="left")
    df = df.merge(pair_wins, on="formation_pair_hash", how="left")
    if not archetype_df.empty:
        df = df.merge(archetype_df, on="match_id", how="left")

    logger.info(f"Built tactical features for {df['home_formation_id'].notna().sum()} matches.")
    return df
