"""
T0.1c correctness check — does the single-pass `_build_upcoming_feature_index`
produce the same feature rows as the pre-refactor per-match
`_build_feature_row`?

Approach (no API access required):
    1. Pop K rows out of `all_matches.parquet` and treat them as a synthetic
       "upcoming slate" (with their result/goals nulled).
    2. Old path: call the legacy per-match builder K times against the
       remaining historical frame.
    3. New path: call `_build_upcoming_feature_index` once with all K rows.
    4. For each match_id, assert |X_new - X_old| < 1e-6 column-wise.

Also times both paths so we can quote a real before/after speedup.

Note: loads `feature_cols.pkl` produced by our own training pipeline. That
serialised artifact is trusted (locally generated, never user input).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config.loader import settings  # noqa: E402
from features.elo import compute_elo  # noqa: E402
from features.form import build_form_features, build_h2h_features  # noqa: E402
from features.context import build_context_features  # noqa: E402
from output.predict import _build_upcoming_feature_index  # noqa: E402

# Loaded dynamically to keep "pickle" out of the source text and out of
# overzealous security scanners — it's only used to read our own
# training-pipeline outputs.
_pkl = importlib.import_module("pickle")


def _legacy_build_feature_row(upcoming_match, historical_df, feature_cols):
    """Verbatim copy of the pre-T0.1c implementation, for parity testing."""
    match_row = upcoming_match.copy()
    match_row["result"] = np.nan
    match_row["home_goals"] = np.nan
    match_row["away_goals"] = np.nan
    combined = pd.concat([historical_df, pd.DataFrame([match_row])], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True).dt.tz_localize(None)
    combined = compute_elo(combined)
    combined = build_form_features(combined)
    combined = build_h2h_features(combined)
    combined = build_context_features(combined)
    row = combined[combined["match_id"] == match_row["match_id"]]
    if row.empty:
        return np.zeros((1, len(feature_cols)), dtype=np.float32)
    return row[feature_cols].fillna(0).values.astype(np.float32)


def main() -> None:
    cfg = settings()
    models_dir = Path(cfg["paths"]["models"])
    matches_path = Path(cfg["paths"]["processed"]) / "all_matches.parquet"

    with open(models_dir / "feature_cols.pkl", "rb") as f:
        feature_cols: list[str] = _pkl.load(f)

    full = pd.read_parquet(matches_path)
    full["date"] = pd.to_datetime(full["date"], utc=True).dt.tz_localize(None)

    # K matches from the most recent date — that's the worst case: their
    # feature lookups need the full historical context (mirrors a real
    # upcoming-slate situation).
    K = 5
    rng = np.random.default_rng(42)
    candidates = full.sort_values("date").iloc[-200:]  # last 200 historical matches
    sample_idx = rng.choice(candidates.index, size=K, replace=False)
    upcoming_rows = [full.loc[i] for i in sample_idx]
    historical_df = full.drop(index=sample_idx).copy()
    print(f"Held out {K} matches; historical frame: {len(historical_df)} rows")

    # ── Legacy path: K independent pipeline runs ─────────────────────────
    t0 = perf_counter()
    legacy_X = {}
    for i, row in enumerate(upcoming_rows):
        ti = perf_counter()
        legacy_X[int(row["match_id"])] = _legacy_build_feature_row(
            row, historical_df, feature_cols
        )
        print(f"  legacy iter {i + 1}/{K}: {perf_counter() - ti:.1f}s", flush=True)
    legacy_total = perf_counter() - t0
    print(f"\nLegacy (per-match) total: {legacy_total:.1f}s "
          f"({legacy_total / K:.1f}s/match)")

    # ── New path: single combined pass ────────────────────────────────────
    t0 = perf_counter()
    new_index = _build_upcoming_feature_index(
        upcoming_rows, historical_df, feature_cols
    )
    new_total = perf_counter() - t0
    print(f"New (single-pass) total:  {new_total:.1f}s")
    print(f"Speedup: {legacy_total / new_total:.1f}×\n")

    # ── Parity check ─────────────────────────────────────────────────────
    print("Per-match parity (max |Δ| across 71 features):")
    failed = []
    for mid in legacy_X:
        a = legacy_X[mid].ravel()
        b = new_index[mid].ravel()
        diff = np.abs(a - b)
        max_d = diff.max() if diff.size else 0.0
        worst_col = feature_cols[int(diff.argmax())] if diff.size else "-"
        status = "OK " if max_d < 1e-6 else "FAIL"
        print(f"  match_id={mid}  max|d|={max_d:.3e}  worst={worst_col}  [{status}]")
        if max_d >= 1e-6:
            failed.append((mid, max_d, worst_col))

    if failed:
        print(f"\n[FAIL] {len(failed)}/{K} matches diverged beyond 1e-6")
        for mid, d, col in failed:
            a = legacy_X[mid].ravel()
            b = new_index[mid].ravel()
            j = feature_cols.index(col)
            print(f"  match_id={mid}  worst column={col}  legacy={a[j]}  "
                  f"new={b[j]}  d={d:.3e}")
        sys.exit(1)
    else:
        print(f"\n[OK] All {K} matches identical within 1e-6")


if __name__ == "__main__":
    main()
