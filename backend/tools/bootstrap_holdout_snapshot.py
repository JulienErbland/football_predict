"""
T2.1 commit 7 — seed the locked holdout snapshot.

Reads ``data/processed/all_matches.parquet``, filters to the holdout
season, sorts match_ids lexicographically as strings, and writes
``data/models/holdout_snapshot.v1.json``.

Run **once** when the holdout is first sealed; subsequent runs require
``--force``, which auto-backs up the existing file to
``data/models/_backup/holdout_snapshot.v1.<old-lock-date>.json`` before
overwriting.

The string-sort discipline (`sorted(str(mid) ...)`) is deterministic
across pandas dtype changes — without it, a future ingestion that
re-types match_ids would silently invalidate every existing snapshot.

CLI usage (from backend/):
    python -m tools.bootstrap_holdout_snapshot
    python -m tools.bootstrap_holdout_snapshot --force
    python -m tools.bootstrap_holdout_snapshot --season 2024 --lock-date 2026-04-29
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from config.loader import settings
from features.build import FEATURE_SCHEMA_VERSION


_SCHEMA_VERSION = "holdout_snapshot.v1"


def _load_holdout_match_ids(processed_dir: Path, season: int) -> list[str]:
    path = processed_dir / "all_matches.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"all_matches.parquet not found at {path}. Run ingestion first."
        )
    df = pd.read_parquet(path)
    season_df = df[df["season"] == season]
    if season_df.empty:
        raise ValueError(f"No matches found for season {season}.")
    return sorted(str(mid) for mid in season_df["match_id"].tolist())


def hash_match_ids(sorted_ids: list[str]) -> str:
    hash_input = ",".join(sorted_ids)
    return "sha256:" + hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def _build_snapshot(season: int, lock_date: str, match_ids: list[str]) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "lock_date": lock_date,
        "season": int(season),
        "n_matches": len(match_ids),
        "match_ids": match_ids,
        "match_ids_sha256": hash_match_ids(match_ids),
    }


def _refuse_message(existing: dict) -> str:
    return (
        "holdout_snapshot.v1.json already exists.\n"
        f"  lock_date: {existing.get('lock_date')}\n"
        f"  hash:      {existing.get('match_ids_sha256')}\n"
        "Re-run with --force to regenerate (existing snapshot will be "
        "backed up to _backup/)."
    )


def _back_up_existing(snapshot_path: Path, existing: dict) -> Path:
    """Move the existing snapshot to _backup/ keyed by its lock_date."""
    backup_dir = snapshot_path.parent / "_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    old_lock = existing.get("lock_date", "unknown")
    backup_path = backup_dir / f"holdout_snapshot.v1.{old_lock}.json"
    shutil.copy2(snapshot_path, backup_path)
    return backup_path


def run(season: int, lock_date: str, force: bool) -> int:
    cfg = settings()
    processed_dir = Path(cfg["paths"]["processed"])
    models_dir = Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = models_dir / "holdout_snapshot.v1.json"

    if snapshot_path.exists():
        with open(snapshot_path) as f:
            existing = json.load(f)
        if not force:
            print(_refuse_message(existing), file=sys.stderr)
            return 1
        backup = _back_up_existing(snapshot_path, existing)
        logger.info(f"Backed up existing snapshot → {backup}")

    match_ids = _load_holdout_match_ids(processed_dir, season)
    snapshot = _build_snapshot(season, lock_date, match_ids)
    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    logger.info(
        f"Sealed {len(match_ids)} match_ids for season {season} "
        f"(hash {snapshot['match_ids_sha256'][:18]}...) → {snapshot_path}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024,
                        help="Holdout season to seal (default: 2024).")
    parser.add_argument("--lock-date", default=date.today().isoformat(),
                        help="ISO date stamp recorded in the snapshot "
                             "(default: today).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing snapshot (auto-backed-up).")
    args = parser.parse_args()
    sys.exit(run(args.season, args.lock_date, args.force))


if __name__ == "__main__":
    main()
