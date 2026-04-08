"""
Elo rating feature builder.

Computes running Elo ratings for all teams sorted chronologically.
K=32 is the standard football Elo update step. Initial rating = 1500 (FIDE-inspired).

Elo-implied probability:
    E_home = 1 / (1 + 10^((R_away - R_home) / 400))
This is the logistic-scale conversion from rating difference to win probability.
We use it as a feature, not as the final prediction, because Elo ignores draw probability
and ignores home-field advantage in its basic form.

All ratings use data up to (but not including) the match being computed —
the shift(1) happens implicitly because we compute ratings from past matches.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


_K = 32
_INITIAL = 1500.0


def _expected(r_a: float, r_b: float) -> float:
    """Expected score for team A vs team B (Elo formula)."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute running Elo ratings for every match.

    Input: matches DataFrame sorted by date (must have columns:
        match_id, date, home_team, away_team, result [0/1/2])

    Returns matches with additional columns:
        home_elo, away_elo, elo_difference, elo_expected_home

    Ratings reflect each team's Elo *before* the match is played.
    """
    df = matches.sort_values("date").copy()
    ratings: dict[str, float] = {}

    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        r_home = ratings.get(home, _INITIAL)
        r_away = ratings.get(away, _INITIAL)

        home_elos.append(r_home)
        away_elos.append(r_away)

        # Update only if result is known (training data)
        if pd.notna(row.get("result")):
            result = int(row["result"])
            # Score for home team: 1=win, 0.5=draw, 0=loss
            s_home = 1.0 if result == 0 else (0.5 if result == 1 else 0.0)
            e_home = _expected(r_home, r_away)
            delta = _K * (s_home - e_home)
            ratings[home] = r_home + delta
            ratings[away] = r_away - delta

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_difference"] = df["home_elo"] - df["away_elo"]
    df["elo_expected_home"] = 1.0 / (
        1.0 + 10.0 ** ((df["away_elo"] - df["home_elo"]) / 400.0)
    )
    return df


def get_current_ratings(matches: pd.DataFrame) -> dict[str, float]:
    """
    Return the most up-to-date Elo rating for every team after processing all matches.

    Used in inference mode to initialise ratings for upcoming match predictions.
    """
    df = matches.sort_values("date")
    ratings: dict[str, float] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("result")):
            continue
        home, away = row["home_team"], row["away_team"]
        r_home = ratings.get(home, _INITIAL)
        r_away = ratings.get(away, _INITIAL)
        result = int(row["result"])
        s_home = 1.0 if result == 0 else (0.5 if result == 1 else 0.0)
        e_home = _expected(r_home, r_away)
        delta = _K * (s_home - e_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return ratings
