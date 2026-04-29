"""
Within-season walk-forward cross-validation splits.

`WalkForwardSplit` produces (train, val) index pairs from a chronologically-
ordered match DataFrame. Each fold validates on a contiguous matchday window
within a single CV-pool season; train uses every CV-pool row dated strictly
before the val window's earliest match. The locked holdout season is removed
from both halves entirely.

The default parametrization `(n_splits_per_season=2, val_window_matchdays=9)`
was selected empirically by `tools/validate_cv_parametrization.py` (T2.1
commit 2): tightest RPS std (0.0129 across 6 folds) with mean RPS 0.2096
inside the ±0.01 sanity band of Phase 1's 0.2069 single-split baseline.

Inputs are validated before splitting. Splits raise:
- :class:`InvalidMatchdayError` if `matchday` is missing or has any value
  ≤ 0 (signals ingestion that never ran the T2.1 commit-1 derivation).
- :class:`InsufficientFoldDataError` if the produced fold count is less than
  `n_splits_per_season × len(cv_pool_seasons)` or any fold has fewer than
  `MIN_VAL_ROWS` validation rows (the design's sanity floor).

Example:
    >>> splitter = WalkForwardSplit()
    >>> for fold_id, (train_idx, val_idx) in enumerate(splitter.split(df)):
    ...     X_tr, y_tr = X[train_idx], y[train_idx]
    ...     X_v, y_v = X[val_idx], y[val_idx]
    ...     # train, predict, evaluate ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from evaluation.exceptions import FootballPredictError


MIN_VAL_ROWS = 100  # design sanity floor — folds smaller than this are invalid


class InvalidMatchdayError(FootballPredictError):
    """`matchday` column is missing, or contains non-positive values."""


class InsufficientFoldDataError(FootballPredictError):
    """Fewer folds were produced than the parametrization promises, or any
    fold has n_val < MIN_VAL_ROWS. Indicates the dataset is too small or
    too sparse to support the requested CV parametrization."""


@dataclass(frozen=True)
class WalkForwardSplit:
    """Walk-forward CV iterator with within-season folds.

    Attributes:
        n_splits_per_season: Number of evenly-spaced val windows per season.
        val_window_matchdays: Width of each val window in matchdays.
        cv_pool_seasons: Tuple of seasons available for both train and val.
        holdout_season: Season excluded from CV entirely (locked holdout).

    The defaults are locked to the empirical winner from commit 2; changing
    them requires re-running ``tools/validate_cv_parametrization.py``.
    """

    n_splits_per_season: int = 2
    val_window_matchdays: int = 9
    cv_pool_seasons: tuple[int, ...] = (2021, 2022, 2023)
    holdout_season: int = 2024

    def __post_init__(self) -> None:
        if self.n_splits_per_season < 1:
            raise ValueError("n_splits_per_season must be >= 1")
        if self.val_window_matchdays < 1:
            raise ValueError("val_window_matchdays must be >= 1")
        if self.holdout_season in self.cv_pool_seasons:
            raise ValueError(
                f"holdout_season {self.holdout_season} cannot be in cv_pool_seasons"
            )

    @property
    def expected_n_folds(self) -> int:
        return self.n_splits_per_season * len(self.cv_pool_seasons)

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) pairs as positional indices into ``df``.

        ``df`` is sorted by date internally; the yielded indices reference
        the *sorted* positions, so callers must sort before indexing into
        their feature arrays. Use :meth:`fold_specs` to inspect fold metadata
        without consuming the iterator.
        """
        for _, train_idx, val_idx in self._iter_with_meta(df):
            yield train_idx, val_idx

    def fold_specs(self, df: pd.DataFrame) -> list[dict]:
        """Return per-fold metadata as plain dicts (for logging / reports).

        Keys: ``fold_id``, ``val_season``, ``val_matchday_range``,
        ``val_start_date``, ``n_train``, ``n_val``.
        """
        return [
            {
                "fold_id": meta["fold_id"],
                "val_season": meta["val_season"],
                "val_matchday_range": meta["val_matchday_range"],
                "val_start_date": meta["val_start_date"],
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            }
            for meta, train_idx, val_idx in self._iter_with_meta(df)
        ]

    def _iter_with_meta(
        self, df: pd.DataFrame
    ) -> list[tuple[dict, np.ndarray, np.ndarray]]:
        self._validate_matchday(df)
        sorted_df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
        pool_mask = sorted_df["season"].isin(self.cv_pool_seasons)
        pool = sorted_df[pool_mask]
        if pool.empty:
            raise InsufficientFoldDataError(
                f"No rows in CV pool seasons {self.cv_pool_seasons}"
            )
        pool_dates = pd.to_datetime(pool["date"])

        results: list[tuple[dict, np.ndarray, np.ndarray]] = []
        fold_id = 0
        for season in self.cv_pool_seasons:
            season_rows = pool[pool["season"] == season]
            if season_rows.empty:
                continue
            # Use min-across-leagues season length so windows fit every league.
            max_md = int(season_rows.groupby("league")["matchday"].max().min())
            for s_md, e_md in self._val_windows(max_md):
                in_window = (
                    (pool["season"] == season)
                    & (pool["matchday"] >= s_md)
                    & (pool["matchday"] <= e_md)
                )
                if not in_window.any():
                    continue
                val_start_date = pool_dates[in_window].min()
                train_mask = pool_dates < val_start_date
                val_idx = pool.index[in_window].to_numpy()
                train_idx = pool.index[train_mask].to_numpy()
                meta = {
                    "fold_id": fold_id,
                    "val_season": int(season),
                    "val_matchday_range": (int(s_md), int(e_md)),
                    "val_start_date": val_start_date.isoformat(),
                }
                results.append((meta, train_idx, val_idx))
                fold_id += 1

        self._validate_folds(results)
        return results

    def _val_windows(self, max_md: int) -> list[tuple[int, int]]:
        """Evenly-spaced (start, end) matchday ranges; warmup = val_window."""
        vw = self.val_window_matchdays
        earliest = vw + 1
        latest = max_md - vw + 1
        if latest < earliest:
            return []
        if self.n_splits_per_season == 1:
            return [(latest, min(latest + vw - 1, max_md))]
        step = (latest - earliest) / (self.n_splits_per_season - 1)
        starts = sorted({
            round(earliest + i * step) for i in range(self.n_splits_per_season)
        })
        return [(s, min(s + vw - 1, max_md)) for s in starts]

    def _validate_matchday(self, df: pd.DataFrame) -> None:
        if "matchday" not in df.columns:
            raise InvalidMatchdayError(
                "DataFrame is missing 'matchday' column — run ingestion "
                "with T2.1 commit-1 matchday derivation."
            )
        if (df["matchday"] <= 0).any():
            n_bad = int((df["matchday"] <= 0).sum())
            raise InvalidMatchdayError(
                f"{n_bad} rows have matchday <= 0 — "
                "ingestion did not derive matchday correctly."
            )

    def _validate_folds(
        self, results: list[tuple[dict, np.ndarray, np.ndarray]]
    ) -> None:
        if len(results) < self.expected_n_folds:
            raise InsufficientFoldDataError(
                f"Only {len(results)} of {self.expected_n_folds} folds materialised — "
                "CV pool may be missing a season or matchday windows didn't fit."
            )
        for meta, _train, val in results:
            if len(val) < MIN_VAL_ROWS:
                raise InsufficientFoldDataError(
                    f"Fold {meta['fold_id']} (season={meta['val_season']}, "
                    f"mds={meta['val_matchday_range']}) has n_val={len(val)} "
                    f"< {MIN_VAL_ROWS} sanity floor."
                )

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise TypeError("WalkForwardSplit is not directly iterable; call .split(df).")
