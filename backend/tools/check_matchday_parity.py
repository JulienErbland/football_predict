"""
Parity check: CSV-derived matchdays vs football-data.org API matchdays.

The CSV ingestion path (football_data.co.uk) does not publish a `matchday`
field, so we derive it from per-team chronological order in
:func:`backend.ingestion.football_data_csv.derive_matchdays`. This script
re-fetches a sample of matches from the football-data.org API (which *does*
publish `matchday`) and compares the two.

Tolerance: zero mismatches expected on regular fixtures. The script aborts
with exit code 1 if the mismatch rate exceeds ``--max-mismatch-rate``
(default 5%) and prints the offending rows for manual review.

CLI usage (from backend/):
    python -m tools.check_matchday_parity                       # default: PL,PD,BL1,SA,FL1 × 2024
    python -m tools.check_matchday_parity --leagues PL,SA       # subset of leagues
    python -m tools.check_matchday_parity --season 2023         # different season
    python -m tools.check_matchday_parity --max-mismatch-rate 0 # strict zero-mismatch mode

Requires a valid ``football_data`` API key in config (free tier OK; 5 calls
at 6.5s rate-limit ≈ 35s wallclock for the default run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from config.loader import settings
from ingestion.football_data import FootballDataClient


def _load_csv_matches(league: str, season: int) -> pd.DataFrame:
    """Load CSV-derived matches for one (league, season) from the processed
    parquet — the canonical model-input file. Empty if file or rows missing."""
    cfg = settings()
    path = Path(cfg["paths"]["processed"]) / "all_matches.parquet"
    if not path.exists():
        logger.warning(f"Processed parquet missing: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "matchday" not in df.columns:
        raise RuntimeError(f"{path} has no 'matchday' column — run ingestion first.")
    return df[(df["league"] == league) & (df["season"] == season)].copy()


def _join_and_compare(csv_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join on (home_team, away_team, date) and return per-match comparison.

    Both sides go through ``normalize_columns`` so team names align. Date is
    truncated to the day on both sides to absorb the API's UTC timestamp
    versus the CSV's local-date representation.
    """
    csv_keyed = csv_df.assign(_date=pd.to_datetime(csv_df["date"]).dt.date)[
        ["home_team", "away_team", "_date", "matchday"]
    ].rename(columns={"matchday": "matchday_csv"})
    api_keyed = api_df.assign(_date=pd.to_datetime(api_df["date"]).dt.date)[
        ["home_team", "away_team", "_date", "matchday"]
    ].rename(columns={"matchday": "matchday_api"})
    return csv_keyed.merge(api_keyed, on=["home_team", "away_team", "_date"], how="inner")


def run(leagues: list[str], season: int, max_mismatch_rate: float) -> int:
    client = FootballDataClient()
    total_compared = 0
    total_mismatches = 0
    all_mismatches: list[pd.DataFrame] = []

    for league in leagues:
        csv_df = _load_csv_matches(league, season)
        if csv_df.empty:
            logger.warning(f"Skipping {league} {season}: no CSV data.")
            continue

        logger.info(f"Fetching API matches for {league} {season} (rate-limited ~6.5s)...")
        try:
            api_df = client.fetch_matches(league, season)
        except Exception as e:
            logger.error(f"API fetch failed for {league} {season}: {e} — skipping.")
            continue
        if api_df.empty:
            logger.warning(f"API returned no matches for {league} {season}.")
            continue

        joined = _join_and_compare(csv_df, api_df)
        if joined.empty:
            logger.warning(f"No matches joined for {league} {season} — team-name drift?")
            continue

        mism = joined[joined["matchday_csv"] != joined["matchday_api"]].copy()
        mism["league"] = league
        mism["season"] = season

        n = len(joined)
        m = len(mism)
        rate = m / n if n else 0.0
        logger.info(
            f"{league} {season}: joined {n} matches, "
            f"{m} mismatch{'es' if m != 1 else ''} ({rate:.1%})"
        )

        total_compared += n
        total_mismatches += m
        if m:
            all_mismatches.append(mism)

    if total_compared == 0:
        logger.error("No matches compared — check league/season args and API key.")
        return 2

    overall_rate = total_mismatches / total_compared
    logger.info(
        f"Overall: {total_mismatches}/{total_compared} mismatches ({overall_rate:.2%}); "
        f"threshold = {max_mismatch_rate:.2%}"
    )

    if all_mismatches:
        combined = pd.concat(all_mismatches, ignore_index=True)
        logger.info("Mismatched fixtures:")
        for _, row in combined.iterrows():
            logger.info(
                f"  {row['league']} {row['season']} {row['_date']} "
                f"{row['home_team']} vs {row['away_team']}: "
                f"csv={row['matchday_csv']} api={row['matchday_api']}"
            )

    if overall_rate > max_mismatch_rate:
        logger.error(
            f"Mismatch rate {overall_rate:.2%} exceeds threshold "
            f"{max_mismatch_rate:.2%} — aborting."
        )
        return 1

    logger.info("Parity check passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", default="PL,PD,BL1,SA,FL1",
                        help="Comma-separated league codes (default: all five).")
    parser.add_argument("--season", type=int, default=2024,
                        help="Season start year to sample (default: 2024).")
    parser.add_argument("--max-mismatch-rate", type=float, default=0.05,
                        help="Abort if mismatch rate exceeds this (default: 0.05).")
    args = parser.parse_args()

    leagues = [l.strip() for l in args.leagues.split(",") if l.strip()]
    sys.exit(run(leagues, args.season, args.max_mismatch_rate))


if __name__ == "__main__":
    main()
