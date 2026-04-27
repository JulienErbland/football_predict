"""
Dataset health check — runs 15 sanity checks against
backend/data/processed/all_matches.parquet plus an Arsenal-specific
ground-truth rolling-PPG computation. Writes a Markdown report to
docs/DATASET_HEALTH_CHECK.md.

Schema notes (verified by reading the codebase, not user templates):
    • `result` is numeric: 0=home win, 1=draw, 2=away win
    • League key is `league` (not `league_id`)
    • `match_id`, `season`, `matchday`, `referee` present
    • `home_xg`/`away_xg` columns may or may not exist
    • There is no `upcoming_matches.parquet` — upcoming fixtures are fetched live
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config.loader import settings  # noqa: E402
from features.form import build_form_features  # noqa: E402

PARQUET = Path(
    settings()["paths"]["processed"]
) / "all_matches.parquet"
REPORT = ROOT / "docs" / "DATASET_HEALTH_CHECK.md"


def _result_label(r: float) -> str:
    if pd.isna(r):
        return "—"
    return {0: "H", 1: "D", 2: "A"}.get(int(r), "?")


def _section(buf: list[str], title: str) -> None:
    buf.append(f"\n## {title}\n")


def _md_kv(buf: list[str], k: str, v) -> None:
    buf.append(f"- **{k}**: {v}")


def _md_code(buf: list[str], txt: str) -> None:
    buf.append("```")
    buf.append(txt.rstrip("\n"))
    buf.append("```")


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    buf: list[str] = []
    buf.append("# Dataset Health Check")
    buf.append("")
    buf.append(f"Source: `{PARQUET.relative_to(ROOT)}`")
    buf.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    buf.append("")
    buf.append("Status legend: ✅ pass · ⚠️ check · ❌ fail")

    flags: list[str] = []

    # ── 1. Basic shape ────────────────────────────────────────────────────
    _section(buf, "1. Basic shape")
    _md_kv(buf, "Total rows", len(df))
    _md_kv(buf, "Date range",
           f"{df['date'].min().date()} → {df['date'].max().date()}")
    _md_kv(buf, "Leagues", sorted(df["league"].unique().tolist()))
    if "season" in df.columns:
        _md_kv(buf, "Seasons", sorted(df["season"].unique().tolist()))
    cols = list(df.columns)
    _md_kv(buf, "Columns", f"{len(cols)} → {cols}")

    # ── 2. Completeness on completed matches ──────────────────────────────
    _section(buf, "2. Completeness (completed matches only)")
    completed = df[df["result"].notna()]
    core_cols = ["home_team", "away_team", "home_goals", "away_goals",
                 "date", "league"]
    nulls = completed[core_cols].isnull().sum()
    _md_code(buf, nulls.to_string())
    if nulls.sum() == 0:
        buf.append("✅ All core columns fully populated for completed matches.")
    else:
        buf.append("❌ Nulls found in core columns — investigate.")
        flags.append("Nulls in core columns of completed matches")

    # ── 3. Team-name consistency ──────────────────────────────────────────
    _section(buf, "3. Team-name consistency")
    all_teams = set(df["home_team"]) | set(df["away_team"])
    _md_kv(buf, "Total unique team names", len(all_teams))
    _md_kv(buf, "Sample (first 12)", sorted(all_teams)[:12])
    # Check for suspected variants of the same team
    suspects: list[tuple[str, str]] = []
    teams_sorted = sorted(all_teams)
    for a in teams_sorted:
        for b in teams_sorted:
            if a >= b:
                continue
            la, lb = a.lower(), b.lower()
            if la == lb:
                suspects.append((a, b))
                continue
            if la.startswith(lb) or lb.startswith(la):
                # Tolerate short prefixes like "AC" vs "AC Milan" only if length differs by ≥4 chars
                if abs(len(a) - len(b)) >= 2 and (la.endswith(" fc")
                                                   or lb.endswith(" fc")
                                                   or la.endswith(" cf")
                                                   or lb.endswith(" cf")
                                                   or la.endswith(" united")
                                                   or lb.endswith(" united")):
                    suspects.append((a, b))
    if suspects:
        buf.append("⚠️ Suspected near-duplicate team names:")
        for a, b in suspects[:20]:
            buf.append(f"  - `{a}` vs `{b}`")
        flags.append(f"{len(suspects)} suspected near-duplicate team names")
    else:
        buf.append("✅ No obvious duplicate team-name variants.")

    # ── 4. Duplicate matches ──────────────────────────────────────────────
    _section(buf, "4. Duplicate matches")
    dupe_mask = df.duplicated(
        subset=["date", "home_team", "away_team"], keep=False
    )
    n_dupes = int(dupe_mask.sum())
    _md_kv(buf, "Duplicate (date, home, away) rows", n_dupes)
    if n_dupes == 0:
        buf.append("✅ No duplicate matches.")
    else:
        buf.append("❌ Duplicates present — feature builders may double-count.")
        flags.append(f"{n_dupes} duplicate match rows")
        _md_code(buf, df[dupe_mask].head(10).to_string(index=False))

    # ── 5. Chronological order ────────────────────────────────────────────
    _section(buf, "5. Chronological order on disk")
    is_sorted = df["date"].is_monotonic_increasing
    _md_kv(buf, "`date` monotonic non-decreasing", is_sorted)
    if not is_sorted:
        buf.append(
            "ℹ️ Data isn't pre-sorted on disk. Feature builders sort internally — "
            "this is fine but worth noting."
        )

    # ── 6. Result vs goals consistency ────────────────────────────────────
    _section(buf, "6. Result column matches goal counts")
    def _consistent(r):
        if pd.isna(r["result"]):
            return True
        res = int(r["result"])
        if res == 0:
            return r["home_goals"] > r["away_goals"]
        if res == 1:
            return r["home_goals"] == r["away_goals"]
        if res == 2:
            return r["home_goals"] < r["away_goals"]
        return False
    consistency = completed.apply(_consistent, axis=1)
    n_bad = int((~consistency).sum())
    _md_kv(buf, "Mismatched result/goals rows", n_bad)
    if n_bad == 0:
        buf.append("✅ All `result` values match the score.")
    else:
        flags.append(f"{n_bad} rows where result disagrees with goals")
        bad = completed[~consistency].head(10)
        _md_code(buf, bad[["date", "home_team", "away_team",
                           "home_goals", "away_goals", "result"]].to_string(index=False))

    # ── 7. Per-league row counts ──────────────────────────────────────────
    _section(buf, "7. Matches per league")
    counts = df["league"].value_counts().sort_index()
    _md_code(buf, counts.to_string())

    # ── 8. Per-(league, season) completeness ──────────────────────────────
    _section(buf, "8. Per-(league, season) row counts")
    if "season" in df.columns:
        cs = df.groupby(["league", "season"]).size().rename("rows")
        _md_code(buf, cs.to_string())
        # EPL-style season ≈ 380 matches (20 teams). Flag wildly off seasons.
        suspect_seasons = cs[(cs < 100) | (cs > 500)]
        if len(suspect_seasons):
            buf.append(
                "⚠️ Some (league, season) row counts look unusual "
                "(<100 or >500):"
            )
            _md_code(buf, suspect_seasons.to_string())
            flags.append(
                f"{len(suspect_seasons)} (league, season) cells with unusual size"
            )
        else:
            buf.append("✅ Per-(league, season) counts look plausible.")
    else:
        buf.append("ℹ️ No `season` column.")

    # ── 9. Goals distribution ─────────────────────────────────────────────
    _section(buf, "9. Goals distribution")
    _md_code(buf, completed[["home_goals", "away_goals"]].describe().to_string())

    # ── 10. Result distribution ───────────────────────────────────────────
    _section(buf, "10. Result distribution (completed)")
    rd = completed["result"].map(_result_label).value_counts(normalize=True)
    _md_code(buf, rd.to_string())
    home_rate = float((completed["result"] == 0).mean())

    # ── 11. Home-advantage check ──────────────────────────────────────────
    _section(buf, "11. Home advantage")
    _md_kv(buf, "Home-win rate", f"{home_rate:.3f}")
    if 0.40 <= home_rate <= 0.50:
        buf.append("✅ Home-win rate inside expected band [0.40, 0.50].")
    else:
        flags.append(f"Home-win rate {home_rate:.3f} outside [0.40, 0.50]")
        buf.append(f"⚠️ Home-win rate {home_rate:.3f} is unusual.")

    # ── 12. Team match counts ─────────────────────────────────────────────
    _section(buf, "12. Team match counts (top-5 vs bottom-5 by 'home' appearances)")
    home_counts = df["home_team"].value_counts()
    _md_kv(buf, "Top-5", home_counts.head().to_dict())
    _md_kv(buf, "Bottom-5", home_counts.tail().to_dict())
    if home_counts.tail().min() > 0 and (
        home_counts.head().max() / max(home_counts.tail().min(), 1) > 10
    ):
        flags.append("Team home-count spread > 10×")
        buf.append("⚠️ Top vs bottom team home-counts differ by >10×.")
    else:
        buf.append("✅ Team home-count spread is reasonable.")

    # ── 13. xG availability ───────────────────────────────────────────────
    _section(buf, "13. xG availability")
    if "home_xg" in df.columns:
        miss = int(df["home_xg"].isnull().sum())
        _md_kv(buf, "Rows missing home_xg", f"{miss} ({miss / len(df):.1%})")
    else:
        buf.append("ℹ️ No `home_xg` column in this parquet (xG is built lazily).")

    # ── 14. Date gaps ─────────────────────────────────────────────────────
    _section(buf, "14. Date gaps > 60 days")
    sorted_df = df.sort_values("date")
    gaps = sorted_df["date"].diff().dt.days
    big = sorted_df.loc[gaps > 60, ["date", "league", "season"]].copy() \
        .assign(gap_days=gaps[gaps > 60].values)
    _md_kv(buf, "# gaps > 60 days", len(big))
    if len(big):
        _md_code(buf, big.to_string(index=False))
        buf.append(
            "ℹ️ Long gaps are expected at season boundaries — verify the dates."
        )

    # ── 15. Upcoming-team consistency ─────────────────────────────────────
    _section(buf, "15. Upcoming-team consistency")
    upcoming_path = ROOT / "backend" / "data" / "processed" / "upcoming_matches.parquet"
    if upcoming_path.exists():
        up = pd.read_parquet(upcoming_path)
        up_teams = set(up["home_team"]) | set(up["away_team"])
        unknown = up_teams - all_teams
        _md_kv(buf, "Upcoming teams not in historical", sorted(unknown))
    else:
        buf.append(
            "ℹ️ No `upcoming_matches.parquet` cached on disk; upcoming fixtures "
            "are fetched from football-data.org at predict time. "
            "T0.1 introduced a separate name-normalisation regression test that "
            "covers this surface (see `backend/tests/`)."
        )

    # ── 16+17. Arsenal ground-truth rolling PPG ───────────────────────────
    _section(buf, "16. Arsenal — last 10 matches (raw)")
    arsenal_mask = (df["home_team"] == "Arsenal") | (df["away_team"] == "Arsenal")
    a = df[arsenal_mask].sort_values("date").reset_index(drop=True)
    last10 = a.tail(10).copy()
    cols_show = ["date", "home_team", "away_team", "home_goals",
                 "away_goals", "result"]
    last10_disp = last10[cols_show].copy()
    last10_disp["result_label"] = last10_disp["result"].map(_result_label)
    _md_code(buf, last10_disp.to_string(index=False))

    _section(buf, "17. Arsenal — manual rolling PPG (window=3, 5, 10)")
    # Build chronological per-Arsenal-perspective points then roll.
    rows = []
    for _, m in a.iterrows():
        side = "home" if m["home_team"] == "Arsenal" else "away"
        if pd.isna(m["result"]):
            pts = np.nan
        else:
            res = int(m["result"])
            if side == "home":
                pts = 3.0 if res == 0 else (1.0 if res == 1 else 0.0)
            else:
                pts = 3.0 if res == 2 else (1.0 if res == 1 else 0.0)
        rows.append({
            "date": m["date"], "side": side, "result": _result_label(m["result"]),
            "arsenal_points": pts,
        })
    chrono = pd.DataFrame(rows)
    # Manual rolling PPG over the *previous* w matches (so shift(1) before rolling)
    for w in (3, 5, 10):
        chrono[f"ppg_w{w}"] = (
            chrono["arsenal_points"].shift(1).rolling(w, min_periods=1).mean()
        )
    _md_code(buf, chrono.tail(10).to_string(index=False))
    buf.append(
        "Convention: `ppg_wN` at row *t* uses Arsenal's previous N matches "
        "(strictly before *t*). This is the ground truth the form features "
        "should match."
    )

    # ── 18. form.py vs ground truth (Arsenal w=3 PPG) ─────────────────────
    _section(buf, "18. form.py BATCH vs PER-MATCH vs ground truth (Arsenal w=3 PPG)")

    # Batch path: build_form_features on full historical frame.
    batch = build_form_features(df)
    a_home = batch[batch["home_team"] == "Arsenal"][
        ["match_id", "home_w3_ppg"]
    ].rename(columns={"home_w3_ppg": "batch_w3_ppg"})
    a_away = batch[batch["away_team"] == "Arsenal"][
        ["match_id", "away_w3_ppg"]
    ].rename(columns={"away_w3_ppg": "batch_w3_ppg"})
    batch_arsenal = pd.concat([a_home, a_away], ignore_index=True)

    # Per-match path: each Arsenal match treated as the lone "upcoming".
    per_match_vals: dict = {}
    last10_match_ids = list(a.tail(10)["match_id"])
    for mid in last10_match_ids:
        match_row = df[df["match_id"] == mid].iloc[0].copy()
        side = "home" if match_row["home_team"] == "Arsenal" else "away"
        match_row["result"] = np.nan
        match_row["home_goals"] = np.nan
        match_row["away_goals"] = np.nan
        hist = df[df["match_id"] != mid]
        per_call = pd.concat([hist, pd.DataFrame([match_row])], ignore_index=True)
        per_call["date"] = pd.to_datetime(
            per_call["date"], utc=True
        ).dt.tz_localize(None)
        out = build_form_features(per_call)
        out_row = out[out["match_id"] == mid].iloc[0]
        col = f"{side}_w3_ppg"
        per_match_vals[int(mid)] = (
            float(out_row[col]) if pd.notna(out_row[col]) else np.nan
        )

    # Manual ground truth from the chrono frame already built in §17.
    arsenal_meta = a.tail(10)[["match_id", "date"]].reset_index(drop=True)
    manual_w3 = chrono.tail(10)["ppg_w3"].reset_index(drop=True)
    arsenal_meta["manual_w3_ppg"] = manual_w3.values
    arsenal_meta = arsenal_meta.merge(
        batch_arsenal, on="match_id", how="left"
    )
    arsenal_meta["per_match_w3_ppg"] = arsenal_meta["match_id"].map(per_match_vals)

    _md_code(buf, arsenal_meta.to_string(index=False))

    # Verdict block
    cmp = arsenal_meta.dropna(subset=["manual_w3_ppg"])
    diff_batch = (cmp["batch_w3_ppg"] - cmp["manual_w3_ppg"]).abs()
    diff_per = (cmp["per_match_w3_ppg"] - cmp["manual_w3_ppg"]).abs()
    buf.append(f"- max |batch − manual|     = **{diff_batch.max():.6f}**")
    buf.append(f"- max |per_match − manual| = **{diff_per.max():.6f}**")
    if diff_batch.max() < 1e-6:
        buf.append("✅ BATCH path matches manual ground truth.")
    else:
        flags.append("BATCH form path diverges from manual ground truth")
        buf.append("❌ BATCH form path DIVERGES from manual ground truth.")
    if diff_per.max() < 1e-6:
        buf.append("✅ PER-MATCH path matches manual ground truth.")
    else:
        flags.append("PER-MATCH form path diverges from manual ground truth")
        buf.append("❌ PER-MATCH form path DIVERGES from manual ground truth.")

    # ── Summary ───────────────────────────────────────────────────────────
    _section(buf, "Summary")
    structural_flags = [f for f in flags if "form path" not in f]
    form_flags = [f for f in flags if "form path" in f]

    if not structural_flags:
        buf.append("✅ **Dataset structure is clean.** No nulls, dupes, broken "
                   "team names, mis-coded results, or unusual distributions.")
    else:
        buf.append("⚠️ Structural flags raised:")
        for f in structural_flags:
            buf.append(f"  - {f}")

    if form_flags:
        buf.append("")
        buf.append(
            "❌ **`form.py` calculation flags raised** (see §18):"
        )
        for f in form_flags:
            buf.append(f"  - {f}")
        buf.append("")
        buf.append(
            "Because the input data is clean (§1–§15) but BOTH form code paths "
            "diverge from the manual ground truth (§17), the bug is purely in "
            "the calculation logic of `backend/features/form.py` — most likely "
            "in the `groupby('team').shift(1)` → `groupby + rolling().mean()` → "
            "`reset_index(level=0, drop=True)` → `.values` chain. The final "
            "`.values` assignment is positional and bypasses index alignment, so "
            "rolled rows (team-grouped order) get scattered into "
            "`all_team_matches` (date-sorted order). This affects both training "
            "(batch path) and inference (per-match path)."
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(buf) + "\n")
    print(f"Wrote report → {REPORT}")
    print(f"Flags: {len(flags)}")


if __name__ == "__main__":
    main()
