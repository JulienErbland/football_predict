"""
T0.1b — cProfile harness for the predict.py hot path.

Profiles `_build_feature_row` against a representative upcoming-match shape on
the real 7156-row historical frame, repeated 50× to mirror the slate size
that took ~13 min in the previous session. Network-dependent calls
(`FootballDataClient`, `OddsFetcher`) are intentionally bypassed — the goal
is to expose the per-match feature-rebuild cost, which is the dominant term.

Usage:
    python tools/profile_predict.py
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config.loader import settings  # noqa: E402
from output.predict import _build_feature_row, _load_feature_cols  # noqa: E402


def _representative_upcoming(historical_df: pd.DataFrame) -> pd.Series:
    """Use the last historical row as a stand-in for an upcoming match shape."""
    sample = historical_df.iloc[-1].copy()
    sample["match_id"] = 9999999
    sample["date"] = pd.Timestamp.utcnow()
    sample["result"] = None
    sample["home_goals"] = None
    sample["away_goals"] = None
    return sample


def main() -> None:
    cfg = settings()
    models_dir = Path(cfg["paths"]["models"])
    feature_cols = _load_feature_cols(models_dir)

    historical_df = pd.read_parquet(
        Path(cfg["paths"]["processed"]) / "all_matches.parquet"
    )
    print(f"Historical rows: {len(historical_df)}  features: {len(feature_cols)}")

    upcoming = _representative_upcoming(historical_df)

    # Warm-up so import-side work doesn't pollute the per-call cost.
    print("warmup …", flush=True)
    t_warm = perf_counter()
    _build_feature_row(upcoming, historical_df, feature_cols)
    print(f"warmup done in {perf_counter() - t_warm:.1f}s", flush=True)

    # n=3 keeps the harness under 10 minutes — enough samples for cProfile
    # to converge on the bottleneck and yield a credible per-call mean.
    n = 3
    t0 = perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(n):
        ti = perf_counter()
        _build_feature_row(upcoming, historical_df, feature_cols)
        print(f"  iter {i + 1}/{n}: {perf_counter() - ti:.1f}s", flush=True)
    profiler.disable()
    elapsed = perf_counter() - t0

    print(f"\nWall time for {n} _build_feature_row calls: {elapsed:.2f}s "
          f"({elapsed / n * 1000:.0f} ms/call)")

    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(30)
    print(buf.getvalue())

    # Also dump tottime view — exclusive time often pinpoints the real cost.
    buf2 = StringIO()
    pstats.Stats(profiler, stream=buf2).sort_stats("tottime").print_stats(20)
    print("--- by tottime (self time) ---")
    print(buf2.getvalue())


if __name__ == "__main__":
    main()
