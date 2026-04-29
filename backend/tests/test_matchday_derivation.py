"""Tests for matchday derivation in football_data_csv.py (T2.1 Q1).

Three tests:
  * `test_derive_matchdays_synthetic` — unit test of the algorithm itself.
  * `test_matchday_populated_in_dataset` — regression against the actual
    `all_matches.parquet` (catches the matchday=0 bug returning).
  * `test_matchday_ranges_per_league_season` — regression: per-(league,
    season), min matchday is 1 and max equals the league's expected count.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingestion.football_data_csv import derive_matchdays


# Expected matchdays per league. Ligue 1 dropped from 38 to 34 starting in
# season 2023 (2023-24); Bundesliga is always 34; PL/PD/SA are always 38.
_EXPECTED_MAX_MATCHDAY = {
    "PL": lambda season: 38,
    "PD": lambda season: 38,
    "SA": lambda season: 38,
    "BL1": lambda season: 34,
    "FL1": lambda season: 38 if season <= 2022 else 34,
}


def test_derive_matchdays_synthetic() -> None:
    """Unit test: a 4-team mini-league plays a full round-robin; verify
    that each team's nth match is matchday n, and that home/away pairs
    end up on the same matchday for regular fixtures.
    """
    rows = [
        {"date": "2024-08-10", "home_team_id": "A", "away_team_id": "B"},
        {"date": "2024-08-10", "home_team_id": "C", "away_team_id": "D"},
        {"date": "2024-08-17", "home_team_id": "A", "away_team_id": "C"},
        {"date": "2024-08-17", "home_team_id": "B", "away_team_id": "D"},
        {"date": "2024-08-24", "home_team_id": "A", "away_team_id": "D"},
        {"date": "2024-08-24", "home_team_id": "B", "away_team_id": "C"},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["league"] = "TEST"
    df["season"] = 2024

    out = derive_matchdays(df)

    assert list(out["matchday"]) == [1, 1, 2, 2, 3, 3]


def test_derive_matchdays_postponement_desync() -> None:
    """If team A's match is postponed, A and B desync by 1 matchday until
    the postponement is played. The algorithm should resolve via max() —
    A's late-played match becomes A's matchday-2 even though chronologically
    it lands on what is matchday 3 for the rest of the league.
    """
    rows = [
        # Round 1: A vs B normal. C vs D normal.
        {"date": "2024-08-10", "home_team_id": "A", "away_team_id": "B"},
        {"date": "2024-08-10", "home_team_id": "C", "away_team_id": "D"},
        # Round 2: C vs A normal. B vs D postponed (not played yet).
        {"date": "2024-08-17", "home_team_id": "C", "away_team_id": "A"},
        # Round 3: A vs D normal. B vs C normal.
        {"date": "2024-08-24", "home_team_id": "A", "away_team_id": "D"},
        {"date": "2024-08-24", "home_team_id": "B", "away_team_id": "C"},
        # Postponed B vs D played late (after round 3).
        {"date": "2024-08-31", "home_team_id": "B", "away_team_id": "D"},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["league"] = "TEST"
    df["season"] = 2024

    out = derive_matchdays(df).reset_index(drop=True)

    # Matchday assignments under per-team max() rule:
    # A-B (1,1)→1; C-D (1,1)→1; C-A (2,2)→2; A-D (3,2)→3; B-C (2,3)→3;
    # B-D postponed (3,3)→3 (both teams have already played 2 matches before
    # this counts as their 3rd).
    assert list(out["matchday"]) == [1, 1, 2, 3, 3, 3]


def test_matchday_populated_in_dataset(historical_matches) -> None:
    """Regression: every row in all_matches.parquet has a non-zero matchday
    within the league's expected range. Catches the matchday=0 bug returning.
    """
    if historical_matches is None:
        pytest.skip("all_matches.parquet missing — run ingestion first")

    df = historical_matches
    assert (df["matchday"] > 0).all(), (
        f"{(df['matchday'] == 0).sum()} rows have matchday=0; "
        "matchday derivation regressed."
    )

    for league in df["league"].unique():
        league_max = max(
            _EXPECTED_MAX_MATCHDAY[league](s)
            for s in df.loc[df["league"] == league, "season"].unique()
        )
        actual_max = int(df.loc[df["league"] == league, "matchday"].max())
        assert actual_max <= league_max, (
            f"League {league} has matchday {actual_max} > expected ceiling "
            f"{league_max}"
        )


def test_matchday_ranges_per_league_season(historical_matches) -> None:
    """Regression: per-(league, season), min matchday is 1 and max equals
    the league's expected count for that season (38 for PL/PD/SA, 34 for
    BL1; FL1 is 38 through 2022 and 34 from 2023).
    """
    if historical_matches is None:
        pytest.skip("all_matches.parquet missing — run ingestion first")

    df = historical_matches

    for (league, season), group in df.groupby(["league", "season"]):
        expected_max = _EXPECTED_MAX_MATCHDAY[league](int(season))
        actual_min = int(group["matchday"].min())
        actual_max = int(group["matchday"].max())

        assert actual_min == 1, (
            f"{league} {season}: min matchday {actual_min} != 1"
        )
        assert actual_max == expected_max, (
            f"{league} {season}: max matchday {actual_max} != "
            f"expected {expected_max}"
        )
