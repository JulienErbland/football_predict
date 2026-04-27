"""Regression test for the T0.1d form-alignment correctness fix.

The pre-T0.1d ``build_form_features`` ended each rolling-stat assignment with
``rolled[col].values`` which discards the index. Because
``groupby + rolling().mean()`` returns rows in *team-grouped* order while
``all_team_matches`` is *date-sorted*, the positional ``.values`` assignment
scattered each team's rolling values into rows belonging to other teams.

Symptoms (verified empirically — see ``docs/DATASET_HEALTH_CHECK.md`` §17/§18):

    For Arsenal's last 10 PL matches (2024/25 season), the manual ground truth
    rolling PPG (window=3) was [0.667, 1.667, 2.333, 2.333, 1.667, 1.667,
    1.667, 1.333, 0.667, 1.333]. The pre-fix batch path returned values that
    diverged from this ground truth by up to 1.333 PPG — a 67% relative error
    for individual matches.

This test pins Arsenal's manual ground truth and asserts that
``build_form_features`` reproduces it for both the home-side and away-side
columns, so any regression of the alignment bug fails immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from config.loader import settings  # noqa: E402
from features.form import build_form_features  # noqa: E402


# Manual ground truth — produced by walking Arsenal's matches chronologically
# and accumulating points; reproducible via tools/dataset_health_check.py §17.
ARSENAL_LAST10_W3_PPG = [
    ("2025-03-16", 0.6666666666666666),
    ("2025-04-01", 1.6666666666666667),
    ("2025-04-05", 2.3333333333333335),
    ("2025-04-12", 2.3333333333333335),
    ("2025-04-20", 1.6666666666666667),
    ("2025-04-23", 1.6666666666666667),
    ("2025-05-03", 1.6666666666666667),
    ("2025-05-11", 1.3333333333333333),
    ("2025-05-18", 0.6666666666666666),
    ("2025-05-25", 1.3333333333333333),
]


@pytest.fixture(scope="module")
def historical_matches() -> pd.DataFrame:
    """Load the project's historical match table once per module."""
    matches_path = Path(settings()["paths"]["processed"]) / "all_matches.parquet"
    if not matches_path.exists():
        pytest.skip(f"all_matches.parquet missing at {matches_path}")
    df = pd.read_parquet(matches_path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return df


@pytest.fixture(scope="module")
def form_table(historical_matches: pd.DataFrame) -> pd.DataFrame:
    """Run build_form_features once per module (it's the slow part)."""
    return build_form_features(historical_matches)


def _arsenal_w3_ppg_from_form_table(form_table: pd.DataFrame) -> dict[str, float]:
    """Pull Arsenal's w3 PPG from the form table, indexed by ISO date."""
    home_rows = form_table[form_table["home_team"] == "Arsenal"][
        ["date", "home_w3_ppg"]
    ].rename(columns={"home_w3_ppg": "w3_ppg"})
    away_rows = form_table[form_table["away_team"] == "Arsenal"][
        ["date", "away_w3_ppg"]
    ].rename(columns={"away_w3_ppg": "w3_ppg"})
    arsenal = pd.concat([home_rows, away_rows], ignore_index=True)
    arsenal = arsenal.sort_values("date").reset_index(drop=True)
    return {
        d.date().isoformat(): float(v)
        for d, v in zip(arsenal["date"], arsenal["w3_ppg"])
    }


def test_arsenal_w3_ppg_matches_manual_ground_truth(form_table: pd.DataFrame) -> None:
    """
    The form table's w3 PPG for Arsenal's last 10 matches must match the
    manually-computed ground truth at 1e-9. Failure here means the bug is
    back: ``rolled[col].values`` is silently scattering teams again.
    """
    by_date = _arsenal_w3_ppg_from_form_table(form_table)

    mismatches = []
    for iso_date, expected in ARSENAL_LAST10_W3_PPG:
        actual = by_date.get(iso_date)
        if actual is None:
            mismatches.append((iso_date, expected, "MISSING"))
            continue
        if not np.isclose(actual, expected, atol=1e-9):
            mismatches.append((iso_date, expected, actual))

    assert not mismatches, (
        "Arsenal w3 PPG diverged from manual ground truth — alignment bug back?\n"
        + "\n".join(f"  {d}: expected {e:.6f}, got {a}" for d, e, a in mismatches)
    )


def test_form_table_no_extreme_ppg_values(form_table: pd.DataFrame) -> None:
    """
    Sanity bound: PPG ∈ [0, 3] by construction. A correctness regression that
    leaks goal counts into PPG slots would blow past 3 immediately.
    """
    for col in ["home_w3_ppg", "away_w3_ppg",
                "home_w5_ppg", "away_w5_ppg",
                "home_w10_ppg", "away_w10_ppg"]:
        if col not in form_table.columns:
            continue
        s = form_table[col].dropna()
        assert s.min() >= 0.0 - 1e-9, f"{col} min {s.min()} < 0"
        assert s.max() <= 3.0 + 1e-9, f"{col} max {s.max()} > 3"
