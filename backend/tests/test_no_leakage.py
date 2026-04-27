"""Property-based leakage gate for the feature pipeline.

THE INVARIANT
-------------
For every completed match ``M`` with kickoff date ``D``, the feature row built
for ``M`` must not depend on any data from matches with date > ``D``. If it
does, the model is being trained against a feature vector that wouldn't be
available at inference time — silent leakage.

WHY THIS TEST IS THE LOAD-BEARING ONE
-------------------------------------
Every other quality gate (RPS, Brier, Draw F1) is downstream of "features are
honest". If a feature builder peeks at the future, every metric we report is
optimistic by an unknown amount, and Phase 2 improvement deltas measure noise.
T0.1d caught one instance of this anti-pattern in ``form.py`` — this test is
the harness that catches the next one before it ships.

DESIGN
------
Each Hypothesis example builds two frames from the same historical corpus::

    A = H (unchanged)
    B = H with home_goals / away_goals / result randomly replaced for every
        row whose date > cutoff

We run all four feature builders (``compute_elo``, ``build_form_features``,
``build_h2h_features``, ``build_context_features``) on both frames, then
assert that every match with date ≤ cutoff has identical feature values in
A and B. If a builder leaks the future, B's perturbation propagates into
those pre-cutoff rows and the equality fails.

Hypothesis varies the perturbation seed across 5 examples — cheap on top of
the (slow) feature-build step, but enough to surface any leakage mode that
only triggers on specific post-cutoff outcomes.

The whole module is skipped if ``all_matches.parquet`` is missing — keeps CI
green on a fresh checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from features.context import build_context_features  # noqa: E402
from features.elo import compute_elo  # noqa: E402
from features.form import build_form_features, build_h2h_features  # noqa: E402


# Columns that are inputs / metadata, not features. Anything OUTSIDE this set
# was added by a feature builder and must therefore be invariant under future
# perturbations of the historical frame.
_INPUT_COLS = frozenset({
    "match_id", "league", "season", "matchday", "date",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
})


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline as predict.py does it."""
    out = frame.copy()
    out = compute_elo(out)
    out = build_form_features(out)
    out = build_h2h_features(out)
    out = build_context_features(out)
    return out


def _perturb_post_cutoff(
    frame: pd.DataFrame, cutoff: pd.Timestamp, seed: int
) -> pd.DataFrame:
    """Replace home_goals / away_goals / result for every row with date > cutoff.

    The new values are drawn from a Poisson(1.4) / Poisson(1.1) — same shape
    as the synthetic generator in ``tools/pipeline_test.py`` — and the result
    label is recomputed so the row stays self-consistent. If the leakage
    invariant holds, none of these post-cutoff changes should propagate into
    pre-cutoff feature rows.
    """
    out = frame.copy()
    post_mask = out["date"] > cutoff
    n_post = int(post_mask.sum())
    if n_post == 0:
        return out

    rng = np.random.default_rng(seed)
    new_home = rng.poisson(1.4, n_post).astype(int)
    new_away = rng.poisson(1.1, n_post).astype(int)
    new_result = np.where(
        new_home > new_away, 0, np.where(new_home == new_away, 1, 2)
    ).astype(out["result"].dtype if "result" in out else int)

    out.loc[post_mask, "home_goals"] = new_home
    out.loc[post_mask, "away_goals"] = new_away
    out.loc[post_mask, "result"] = new_result
    return out


def _diff_pre_cutoff(
    a: pd.DataFrame, b: pd.DataFrame, cutoff: pd.Timestamp
) -> list[str]:
    """Return the list of feature columns that disagree on any pre-cutoff row.

    Pre-cutoff = matches with date ≤ cutoff. A leakage-free pipeline returns
    an empty list. NaN is treated as equal to NaN (both builders produce NaN
    for first-of-season rows; that's a structural artifact, not leakage).
    """
    pre_a = a[a["date"] <= cutoff].set_index("match_id").sort_index()
    pre_b = b[b["date"] <= cutoff].set_index("match_id").sort_index()

    feature_cols = [c for c in pre_a.columns if c not in _INPUT_COLS]
    bad: list[str] = []
    for col in feature_cols:
        sa, sb = pre_a[col], pre_b[col]
        if sa.dtype.kind in "fc" and sb.dtype.kind in "fc":
            equal = np.isclose(
                sa.fillna(np.nan), sb.fillna(np.nan),
                equal_nan=True, rtol=1e-9, atol=1e-9,
            ).all()
        else:
            equal = (sa.equals(sb))
        if not equal:
            bad.append(col)
    return bad


@pytest.fixture(scope="module")
def cutoff(historical_matches) -> pd.Timestamp:
    """Median completed-match date — splits the corpus roughly in half.

    Picking the median (not the start or end) ensures both pre-cutoff and
    post-cutoff sides have enough rows that the perturbation is non-trivial
    AND that there's enough pre-cutoff history for feature builders to do
    real work.
    """
    if historical_matches is None:
        pytest.skip("all_matches.parquet missing")
    completed = historical_matches.dropna(subset=["result"]).sort_values("date")
    if len(completed) < 100:
        pytest.skip("historical corpus too small for leakage test")
    return completed["date"].iloc[len(completed) // 2]


@pytest.fixture(scope="module")
def features_a(historical_matches) -> pd.DataFrame:
    """Features built on the original, unperturbed historical frame.

    Cached at module scope — building it once costs ~5–10s; rebuilding per
    test would dominate CI runtime.
    """
    if historical_matches is None:
        pytest.skip("all_matches.parquet missing")
    return _build_features(historical_matches)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_features_invariant_under_post_cutoff_perturbation(
    historical_matches, cutoff, features_a, seed: int
) -> None:
    """The core leakage gate.

    For each Hypothesis-generated perturbation seed, build a corrupted frame
    B (post-cutoff goals/results randomised) and assert that the feature
    columns for every pre-cutoff match still match the unperturbed baseline.

    A failure here means the named columns leaked information from after the
    cutoff. The repair is the same shape as T0.1d: find the offending feature
    builder, audit its rolling / aggregation calls for `shift(1)` and
    index-aligned assignment.
    """
    if historical_matches is None:
        pytest.skip("all_matches.parquet missing")

    frame_b = _perturb_post_cutoff(historical_matches, cutoff, seed)
    features_b = _build_features(frame_b)

    leaking_cols = _diff_pre_cutoff(features_a, features_b, cutoff)
    assert not leaking_cols, (
        f"Feature leakage detected (perturbation seed={seed}, cutoff={cutoff}).\n"
        f"These columns changed for pre-cutoff matches when post-cutoff data "
        f"was perturbed — they must not depend on the future:\n  - "
        + "\n  - ".join(leaking_cols)
    )


def test_perturbation_actually_changes_post_cutoff_features(
    historical_matches, cutoff, features_a
) -> None:
    """Sanity: confirm the perturbation is actually doing something.

    If post-cutoff feature columns were ALSO unchanged we'd have a vacuously
    passing leakage test (e.g. perturbation accidentally a no-op). This
    asserts at least one post-cutoff feature column moves under perturbation
    so we know the test is doing real work.
    """
    if historical_matches is None:
        pytest.skip("all_matches.parquet missing")

    frame_b = _perturb_post_cutoff(historical_matches, cutoff, seed=12345)
    features_b = _build_features(frame_b)

    post_a = features_a[features_a["date"] > cutoff].set_index("match_id").sort_index()
    post_b = features_b[features_b["date"] > cutoff].set_index("match_id").sort_index()
    feature_cols = [c for c in post_a.columns if c not in _INPUT_COLS]

    moved = [
        c for c in feature_cols
        if not np.isclose(
            post_a[c].fillna(0).to_numpy(),
            post_b[c].fillna(0).to_numpy(),
            rtol=1e-9, atol=1e-9,
        ).all()
    ]
    assert moved, (
        "Perturbation didn't change any post-cutoff feature column — the "
        "leakage test is vacuously passing. Investigate _perturb_post_cutoff."
    )
