"""
Unit tests for backend.evaluation.splits.WalkForwardSplit.

Covers fold construction, leakage prevention, parametrization knobs, and the
matchday validator. Uses the existing `historical_matches` fixture from
conftest.py — no synthetic builder needed because matchday is now correctly
populated post-T2.1 commit 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.splits import (
    WalkForwardSplit,
    InvalidMatchdayError,
    InsufficientFoldDataError,
)


_DEFAULT_KWARGS = dict(
    n_splits_per_season=2,
    val_window_matchdays=9,
    cv_pool_seasons=(2021, 2022, 2023),
    holdout_season=2024,
)


def test_default_parametrization_locked_to_commit_2_winner():
    """The default (n_splits, vw) pair is the empirical winner from commit 2."""
    splitter = WalkForwardSplit()
    assert splitter.n_splits_per_season == 2
    assert splitter.val_window_matchdays == 9


def test_n_folds_matches_pool_size_times_splits(historical_matches: pd.DataFrame):
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    folds = list(splitter.split(historical_matches))
    assert len(folds) == 6  # 3 CV-pool seasons × 2 splits


@pytest.mark.parametrize("n_splits,vw,expected", [
    (3, 6, 9),
    (4, 5, 12),
    (2, 9, 6),
])
def test_alternative_parametrizations_yield_expected_fold_counts(
    historical_matches: pd.DataFrame, n_splits: int, vw: int, expected: int,
):
    splitter = WalkForwardSplit(
        n_splits_per_season=n_splits,
        val_window_matchdays=vw,
        cv_pool_seasons=(2021, 2022, 2023),
        holdout_season=2024,
    )
    assert len(list(splitter.split(historical_matches))) == expected


def test_train_strictly_precedes_val_chronologically(historical_matches: pd.DataFrame):
    """No leakage: every train row must be dated before every val row in its fold."""
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    df = historical_matches.sort_values("date").reset_index(drop=True)
    for train_idx, val_idx in splitter.split(df):
        train_max_date = df.iloc[train_idx]["date"].max()
        val_min_date = df.iloc[val_idx]["date"].min()
        assert train_max_date < val_min_date, \
            f"Leakage: train max {train_max_date} >= val min {val_min_date}"


def test_holdout_season_never_in_val_set(historical_matches: pd.DataFrame):
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    df = historical_matches.sort_values("date").reset_index(drop=True)
    for _, val_idx in splitter.split(df):
        assert (df.iloc[val_idx]["season"] != 2024).all(), \
            "Holdout season 2024 leaked into val"


def test_holdout_season_never_in_train_set(historical_matches: pd.DataFrame):
    """Holdout is locked away — train uses CV-pool seasons only."""
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    df = historical_matches.sort_values("date").reset_index(drop=True)
    for train_idx, _ in splitter.split(df):
        assert (df.iloc[train_idx]["season"] != 2024).all(), \
            "Holdout season 2024 leaked into train"


def test_n_val_above_sanity_floor(historical_matches: pd.DataFrame):
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    df = historical_matches.sort_values("date").reset_index(drop=True)
    for train_idx, val_idx in splitter.split(df):
        assert len(val_idx) >= 100, f"Fold with n_val={len(val_idx)} below floor"


def test_zero_matchday_raises(historical_matches: pd.DataFrame):
    """Matchday=0 means ingestion never derived; refuse to split."""
    df = historical_matches.copy()
    df.loc[df.index[0], "matchday"] = 0
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    with pytest.raises(InvalidMatchdayError, match="matchday"):
        list(splitter.split(df))


def test_missing_matchday_column_raises(historical_matches: pd.DataFrame):
    df = historical_matches.drop(columns=["matchday"])
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    with pytest.raises(InvalidMatchdayError, match="matchday"):
        list(splitter.split(df))


def test_insufficient_data_raises(historical_matches: pd.DataFrame):
    """Removing 2 of 3 CV seasons should produce <expected folds and raise."""
    df = historical_matches[historical_matches["season"].isin([2023, 2024])].copy()
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    with pytest.raises(InsufficientFoldDataError, match="2 of 6 fold"):
        list(splitter.split(df))


def test_split_indices_are_disjoint(historical_matches: pd.DataFrame):
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    for train_idx, val_idx in splitter.split(historical_matches):
        assert not set(train_idx) & set(val_idx), "train/val index overlap"


def test_all_folds_have_pool_seasons_only_in_val(historical_matches: pd.DataFrame):
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    df = historical_matches.sort_values("date").reset_index(drop=True)
    for _, val_idx in splitter.split(df):
        seasons = set(df.iloc[val_idx]["season"].unique())
        assert seasons.issubset({2021, 2022, 2023}), f"Unexpected val seasons: {seasons}"


def test_fold_definitions_are_introspectable(historical_matches: pd.DataFrame):
    """Public `fold_specs` returns metadata for logging / cv_report."""
    splitter = WalkForwardSplit(**_DEFAULT_KWARGS)
    specs = splitter.fold_specs(historical_matches)
    assert len(specs) == 6
    assert all("val_season" in s and "val_matchday_range" in s for s in specs)
    assert all(s["val_season"] in {2021, 2022, 2023} for s in specs)
    seasons = [s["val_season"] for s in specs]
    assert seasons == sorted(seasons), "Folds should be ordered chronologically"
