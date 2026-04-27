"""Shared pytest fixtures for the backend test suite.

Adds ``backend/`` to ``sys.path`` so individual test modules can
``from features.form import …`` without importing-time path gymnastics, and
exposes a few fixtures (``settings_dict``, ``historical_matches``) that the
heavier tests reuse to avoid re-loading parquet files per test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def settings_dict() -> dict:
    """Project settings dict (model_config.yaml + feature_config.yaml merged)."""
    from config.loader import settings  # noqa: PLC0415 — needs sys.path tweak above
    return settings()


@pytest.fixture(scope="session")
def historical_matches(settings_dict: dict):
    """Load all_matches.parquet once per test session.

    Returns ``None`` (and tests that depend on it should ``pytest.skip``) if the
    file doesn't exist — keeps the suite green on a fresh checkout where the
    historical corpus hasn't been built yet.
    """
    import pandas as pd  # noqa: PLC0415

    matches_path = Path(settings_dict["paths"]["processed"]) / "all_matches.parquet"
    if not matches_path.exists():
        return None
    df = pd.read_parquet(matches_path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return df
