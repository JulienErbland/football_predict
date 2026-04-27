"""
Form and head-to-head feature builder.

Rolling features use shift(1).rolling(window, min_periods=1) to prevent leakage:
    - shift(1) ensures the current match's result is NOT included in the rolling window
    - min_periods=1 means partial windows are allowed (first few matches of a season)

Features computed per team, per side (home/away separately):
    win_rate, draw_rate, loss_rate, ppg, avg_gf, avg_ga, avg_gd, clean_sheet_rate
For windows [3, 5, 10].

H2H features use a 5-year lookback (configurable via feature_config.yaml).

T0.1d alignment fix — `groupby(...).rolling().mean()` returns rows in
team-grouped order (the second level of its MultiIndex preserves the original
positional index, but the row sequence is grouped by team). Assigning the
result with `.values` discards the index and lands each team's stats into
whatever rows happen to sit at the same positions in the date-sorted target —
silently scattering Team A's rolling values into Team B/C/D rows. The fix is
to assign the Series directly, letting pandas align by index. See
`backend/tests/test_form_alignment.py` and `docs/DATASET_HEALTH_CHECK.md`.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def _rolling_team_stats(df: pd.DataFrame, team_col: str, windows: list[int],
                        prefix: str) -> pd.DataFrame:
    """
    Compute rolling stats for a team's perspective (home or away).

    df must have columns: team_col, date, is_win, is_draw, is_loss,
        goals_for, goals_against, clean_sheet.
    Returns df with new columns named {prefix}_w{window}_{stat}.
    """
    df = df.sort_values("date").copy()
    for w in windows:
        shifted = df.groupby(team_col)[["is_win", "is_draw", "is_loss",
                                         "goals_for", "goals_against",
                                         "clean_sheet"]].shift(1)
        rolled = shifted.groupby(df[team_col]).rolling(w, min_periods=1).mean()
        # T0.1d: drop the team level only; KEEP the positional index so the
        # subsequent column assignments align by index, not by row position.
        rolled = rolled.reset_index(level=0, drop=True)
        df[f"{prefix}_w{w}_win_rate"] = rolled["is_win"]
        df[f"{prefix}_w{w}_draw_rate"] = rolled["is_draw"]
        df[f"{prefix}_w{w}_loss_rate"] = rolled["is_loss"]
        df[f"{prefix}_w{w}_ppg"] = rolled["is_win"] * 3 + rolled["is_draw"]
        df[f"{prefix}_w{w}_avg_gf"] = rolled["goals_for"]
        df[f"{prefix}_w{w}_avg_ga"] = rolled["goals_against"]
        df[f"{prefix}_w{w}_avg_gd"] = rolled["goals_for"] - rolled["goals_against"]
        df[f"{prefix}_w{w}_clean_sheet_rate"] = rolled["clean_sheet"]
    return df


def build_form_features(matches: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Build rolling form features for home and away teams.

    Input: all_matches DataFrame with match_id, date, home_team, away_team,
        home_goals, away_goals, result.

    Returns the input DataFrame enriched with form feature columns.
    """
    if windows is None:
        windows = [3, 5, 10]

    df = matches.sort_values("date").copy()

    # ── Build per-team event tables (one row per match per team) ────────────
    home_view = df[["match_id", "date", "home_team", "home_goals", "away_goals", "result"]].copy()
    home_view.columns = ["match_id", "date", "team", "goals_for", "goals_against", "result"]
    home_view["is_win"] = (home_view["result"] == 0).astype(float)
    home_view["is_draw"] = (home_view["result"] == 1).astype(float)
    home_view["is_loss"] = (home_view["result"] == 2).astype(float)
    home_view["clean_sheet"] = (home_view["goals_against"] == 0).astype(float)
    home_view["side"] = "home"

    away_view = df[["match_id", "date", "away_team", "away_goals", "home_goals", "result"]].copy()
    away_view.columns = ["match_id", "date", "team", "goals_for", "goals_against", "result"]
    away_view["is_win"] = (away_view["result"] == 2).astype(float)
    away_view["is_draw"] = (away_view["result"] == 1).astype(float)
    away_view["is_loss"] = (away_view["result"] == 0).astype(float)
    away_view["clean_sheet"] = (away_view["goals_against"] == 0).astype(float)
    away_view["side"] = "away"

    all_team_matches = pd.concat([home_view, away_view], ignore_index=True).sort_values("date")

    # ── Rolling stats for all matches combined (regardless of home/away side) ─
    for w in windows:
        shifted = all_team_matches.groupby("team")[
            ["is_win", "is_draw", "is_loss", "goals_for", "goals_against", "clean_sheet"]
        ].shift(1)
        rolled = (
            shifted.groupby(all_team_matches["team"])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # T0.1d: assign the Series (index-aligned), NOT `.values` (positional).
        # `groupby.rolling.mean()` returns rows ordered by team group; the
        # positional .values previously scattered each team's stats into other
        # teams' rows. Index-aligned assignment puts every value in its own row.
        all_team_matches[f"w{w}_win_rate"] = rolled["is_win"]
        all_team_matches[f"w{w}_draw_rate"] = rolled["is_draw"]
        all_team_matches[f"w{w}_loss_rate"] = rolled["is_loss"]
        all_team_matches[f"w{w}_ppg"] = rolled["is_win"] * 3 + rolled["is_draw"]
        all_team_matches[f"w{w}_avg_gf"] = rolled["goals_for"]
        all_team_matches[f"w{w}_avg_ga"] = rolled["goals_against"]
        all_team_matches[f"w{w}_avg_gd"] = rolled["goals_for"] - rolled["goals_against"]
        all_team_matches[f"w{w}_clean_sheet_rate"] = rolled["clean_sheet"]

    # ── Merge home stats back to match DataFrame ────────────────────────────
    home_stats = all_team_matches[all_team_matches["side"] == "home"].set_index("match_id")
    away_stats = all_team_matches[all_team_matches["side"] == "away"].set_index("match_id")

    stat_cols = [f"w{w}_{s}" for w in windows
                 for s in ["win_rate", "draw_rate", "loss_rate", "ppg",
                            "avg_gf", "avg_ga", "avg_gd", "clean_sheet_rate"]]

    for col in stat_cols:
        if col in home_stats.columns:
            df[f"home_{col}"] = df["match_id"].map(home_stats[col])
        if col in away_stats.columns:
            df[f"away_{col}"] = df["match_id"].map(away_stats[col])

    return df


def build_h2h_features(matches: pd.DataFrame, window_years: int = 5) -> pd.DataFrame:
    """
    Build head-to-head features with a configurable year lookback.

    For each match, looks at the previous `window_years` years of H2H results
    between the same pair of teams (in either direction).

    Adds columns:
        h2h_games, h2h_home_win_rate, h2h_draw_rate, h2h_away_win_rate, h2h_avg_goals
    """
    df = matches.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    h2h_games, h2h_home_win_rate, h2h_draw_rate, h2h_away_win_rate, h2h_avg_goals = (
        [], [], [], [], []
    )

    for idx, row in df.iterrows():
        cutoff = row["date"] - timedelta(days=window_years * 365)
        # Matches between the same pair of teams in either direction
        mask = (
            (df["date"] < row["date"])
            & (df["date"] >= cutoff)
            & (
                ((df["home_team"] == row["home_team"]) & (df["away_team"] == row["away_team"]))
                | ((df["home_team"] == row["away_team"]) & (df["away_team"] == row["home_team"]))
            )
        )
        past = df[mask]
        n = len(past)
        if n == 0:
            h2h_games.append(0)
            h2h_home_win_rate.append(np.nan)
            h2h_draw_rate.append(np.nan)
            h2h_away_win_rate.append(np.nan)
            h2h_avg_goals.append(np.nan)
            continue

        # Reinterpret results from current home team's perspective
        home_wins = 0
        draws = 0
        away_wins = 0
        total_goals = 0
        for _, h2h_row in past.iterrows():
            if h2h_row["home_team"] == row["home_team"]:
                res = h2h_row["result"]
            else:
                # Flip the result (home/away swapped)
                res = {0: 2, 1: 1, 2: 0}.get(h2h_row["result"], np.nan)
            if res == 0:
                home_wins += 1
            elif res == 1:
                draws += 1
            elif res == 2:
                away_wins += 1
            total_goals += h2h_row["home_goals"] + h2h_row["away_goals"]

        h2h_games.append(n)
        h2h_home_win_rate.append(home_wins / n)
        h2h_draw_rate.append(draws / n)
        h2h_away_win_rate.append(away_wins / n)
        h2h_avg_goals.append(total_goals / n)

    df["h2h_games"] = h2h_games
    df["h2h_home_win_rate"] = h2h_home_win_rate
    df["h2h_draw_rate"] = h2h_draw_rate
    df["h2h_away_win_rate"] = h2h_away_win_rate
    df["h2h_avg_goals"] = h2h_avg_goals
    return df
