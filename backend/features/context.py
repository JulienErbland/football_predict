"""
Context feature builder.

Contextual features that don't fit into form/xG/squad categories:
    - Rest days (days since last match)
    - Fixture congestion (matches in last 30 days)
    - Season stage (matchday / max_matchday, normalised 0→1)
    - League position (computed from past results within the season)
    - Relegation pressure flag (bottom 3)
    - Title race flag (top 4)
    - Referee (label-encoded integer)
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger


def _compute_league_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute each team's league position based on all
    completed results BEFORE the match date (no leakage).

    Adds columns: home_league_pos, away_league_pos, position_gap.
    """
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    home_pos_list, away_pos_list = [], []

    for idx, row in df.iterrows():
        past = df[
            (df["league"] == row["league"])
            & (df["season"] == row["season"])
            & (df["date"] < row["date"])
            & df["result"].notna()
        ]
        if past.empty:
            home_pos_list.append(np.nan)
            away_pos_list.append(np.nan)
            continue

        # Build a mini league table from past matches
        teams = set(past["home_team"]) | set(past["away_team"])
        table: dict[str, dict] = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        for _, m in past.iterrows():
            r = m["result"]
            if r == 0:
                table[m["home_team"]]["pts"] += 3
            elif r == 1:
                table[m["home_team"]]["pts"] += 1
                table[m["away_team"]]["pts"] += 1
            else:
                table[m["away_team"]]["pts"] += 3
            table[m["home_team"]]["gd"] += m["home_goals"] - m["away_goals"]
            table[m["away_team"]]["gd"] += m["away_goals"] - m["home_goals"]
            table[m["home_team"]]["gf"] += m["home_goals"]
            table[m["away_team"]]["gf"] += m["away_goals"]

        ranked = sorted(teams, key=lambda t: (
            -table[t]["pts"], -table[t]["gd"], -table[t]["gf"]
        ))
        pos_map = {team: i + 1 for i, team in enumerate(ranked)}
        home_pos_list.append(pos_map.get(row["home_team"], np.nan))
        away_pos_list.append(pos_map.get(row["away_team"], np.nan))

    df["home_league_pos"] = home_pos_list
    df["away_league_pos"] = away_pos_list
    df["position_gap"] = df["home_league_pos"] - df["away_league_pos"]
    n_teams = df.groupby(["league", "season"])["home_team"].transform("nunique")
    df["home_relegation_pressure"] = (df["home_league_pos"] >= n_teams - 2).astype(float)
    df["away_relegation_pressure"] = (df["away_league_pos"] >= n_teams - 2).astype(float)
    df["home_title_race"] = (df["home_league_pos"] <= 4).astype(float)
    df["away_title_race"] = (df["away_league_pos"] <= 4).astype(float)
    return df


_STANDING_COLS = (
    "home_league_pos",
    "away_league_pos",
    "position_gap",
    "home_relegation_pressure",
    "away_relegation_pressure",
    "home_title_race",
    "away_title_race",
)


def build_context_features(
    matches: pd.DataFrame,
    standings_cache: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """
    Build all context features and return enriched DataFrame.

    This builder always succeeds (no external data dependencies).

    `standings_cache` (T0.1c optimisation) — optional dict keyed by match_id
    holding the seven standings-derived columns: home_league_pos,
    away_league_pos, position_gap, home_relegation_pressure,
    away_relegation_pressure, home_title_race, away_title_race. When supplied,
    the slow `_compute_league_positions` call is skipped and the cached
    columns are injected instead. Used by `output.predict` to avoid
    rebuilding standings 50× per slate; for any cache miss the row's
    standings columns are NaN (caller's responsibility to ensure coverage).
    """
    df = matches.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Rest days and fixture congestion ───────────────────────────────────
    # Build a team → sorted list of match dates lookup
    all_team_dates: dict[str, list] = {}
    for _, row in df.iterrows():
        all_team_dates.setdefault(row["home_team"], []).append(row["date"])
        all_team_dates.setdefault(row["away_team"], []).append(row["date"])
    for t in all_team_dates:
        all_team_dates[t] = sorted(all_team_dates[t])

    home_rest, away_rest, home_congestion, away_congestion = [], [], [], []
    for _, row in df.iterrows():
        d = row["date"]
        for team, rest_list, cong_list in [
            (row["home_team"], home_rest, home_congestion),
            (row["away_team"], away_rest, away_congestion),
        ]:
            dates = all_team_dates.get(team, [])
            before = [x for x in dates if x < d]
            if before:
                rest_list.append((d - max(before)).days)
            else:
                rest_list.append(np.nan)
            # Congestion = matches in last 30 days
            cong_list.append(sum(1 for x in before if (d - x).days <= 30))

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_congestion_30d"] = home_congestion
    df["away_congestion_30d"] = away_congestion

    # ── Season stage ───────────────────────────────────────────────────────
    max_matchday = df.groupby(["league", "season"])["matchday"].transform("max")
    df["season_stage"] = df["matchday"] / max_matchday.replace(0, np.nan)

    # ── Referee encoding ───────────────────────────────────────────────────
    unique_refs = df["referee"].dropna().unique()
    ref_map = {r: i for i, r in enumerate(sorted(unique_refs))}
    df["referee_encoded"] = df["referee"].map(ref_map).fillna(-1).astype(int)

    # ── League positions (expensive — computed last) ───────────────────────
    if standings_cache is not None:
        for col in _STANDING_COLS:
            df[col] = df["match_id"].map(lambda mid, _c=col: standings_cache.get(int(mid), {}).get(_c))
    else:
        logger.info("Computing league positions (this may take a few minutes)...")
        df = _compute_league_positions(df)

    logger.info("Built context features.")
    return df


def extract_standings_cache(df_with_standings: pd.DataFrame) -> dict[int, dict]:
    """
    Extract a {match_id: {standings_col: value}} cache from a frame that
    already has the seven standings columns populated (e.g. the output of
    a prior `_compute_league_positions` run).
    """
    cache: dict[int, dict] = {}
    for _, row in df_with_standings.iterrows():
        cache[int(row["match_id"])] = {col: row[col] for col in _STANDING_COLS if col in row}
    return cache
