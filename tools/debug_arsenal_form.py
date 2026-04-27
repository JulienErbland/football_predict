"""
Debug Arsenal's rolling PPG (window=3) computed three ways:

    1. **Vectorized batch** — current `build_form_features` path on the full
       historical frame (no upcoming).
    2. **Manual incremental** — straight Python loop over Arsenal's chronological
       matches. This is the ground-truth definition: PPG over the *previous* w
       completed matches (regardless of home/away side).
    3. **Per-match** — for each of Arsenal's last 10 matches, treat it as the
       "upcoming" match: drop it from history, null its result, append it back,
       run `build_form_features`, read the form value at that match.

We expect (1) and (3) to agree with (2) at every match. If they diverge that's
our bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config.loader import settings  # noqa: E402
from features.form import build_form_features  # noqa: E402

W = 3  # window size for this debug


def manual_ppg(team_matches: pd.DataFrame, team: str, w: int) -> list[float]:
    """
    Pure-Python incremental rolling PPG over the team's previous `w` matches
    (regardless of home/away side). Walks `team_matches` chronologically.
    Returns one float per row (NaN before any prior match exists).
    """
    out: list[float] = []
    prior_pts: list[float] = []  # in chronological order
    for _, row in team_matches.iterrows():
        # Compute rolling PPG from `prior_pts` BEFORE adding this match.
        if not prior_pts:
            out.append(np.nan)
        else:
            window_pts = prior_pts[-w:]
            out.append(sum(window_pts) / len(window_pts))

        # Now record THIS match's points (only if completed).
        if pd.notna(row["result"]):
            res = int(row["result"])
            if row["side"] == "home":
                if res == 0:
                    prior_pts.append(3.0)
                elif res == 1:
                    prior_pts.append(1.0)
                else:
                    prior_pts.append(0.0)
            else:  # away
                if res == 2:
                    prior_pts.append(3.0)
                elif res == 1:
                    prior_pts.append(1.0)
                else:
                    prior_pts.append(0.0)
        else:
            # Upcoming match — no result to add to prior_pts.
            pass
    return out


def main() -> None:
    cfg = settings()
    matches_path = Path(cfg["paths"]["processed"]) / "all_matches.parquet"
    full = pd.read_parquet(matches_path)
    full["date"] = pd.to_datetime(full["date"], utc=True).dt.tz_localize(None)
    full = full.sort_values("date").reset_index(drop=True)

    TEAM = "Arsenal"
    arsenal_mask = (full["home_team"] == TEAM) | (full["away_team"] == TEAM)
    arsenal_all = full[arsenal_mask].sort_values("date").reset_index(drop=True)
    print(f"Arsenal historical matches: {len(arsenal_all)}")

    # ── Build the chronological (team-perspective) frame for manual PPG ────
    rows = []
    for _, m in arsenal_all.iterrows():
        side = "home" if m["home_team"] == TEAM else "away"
        rows.append({
            "match_id": m["match_id"],
            "date": m["date"],
            "side": side,
            "result": m["result"],
        })
    team_chrono = pd.DataFrame(rows)
    manual = manual_ppg(team_chrono, TEAM, W)
    team_chrono["manual_w3_ppg"] = manual

    # ── Run vectorized batch path ─────────────────────────────────────────
    batch = build_form_features(full)
    # Pull Arsenal's home and away rows out separately and merge by match_id
    a_home = batch[batch["home_team"] == TEAM][["match_id", "date", f"home_w{W}_ppg"]]
    a_away = batch[batch["away_team"] == TEAM][["match_id", "date", f"away_w{W}_ppg"]]
    a_home = a_home.rename(columns={f"home_w{W}_ppg": "batch_w3_ppg"})
    a_away = a_away.rename(columns={f"away_w{W}_ppg": "batch_w3_ppg"})
    batch_arsenal = pd.concat([a_home, a_away], ignore_index=True).sort_values("date")

    cmp_df = team_chrono.merge(batch_arsenal[["match_id", "batch_w3_ppg"]],
                               on="match_id", how="left")

    # Show the last 10 Arsenal matches
    last_10 = cmp_df.tail(10).copy()
    print("\n── Last 10 Arsenal matches (BATCH vs MANUAL w=3 PPG) ──────────")
    print(last_10[["date", "side", "result", "manual_w3_ppg",
                   "batch_w3_ppg"]].to_string(index=False))

    diff = (last_10["manual_w3_ppg"].fillna(-999)
            - last_10["batch_w3_ppg"].fillna(-999)).abs()
    max_diff = diff[diff < 100].max()  # exclude the -999 sentinel diffs
    print(f"\nMax |manual − batch| over last 10 (excl. NaN): {max_diff:.6f}")

    # ── Per-match: simulate upcoming for each of Arsenal's last 10 matches ─
    print("\n── Per-match: each of Arsenal's last 10 treated as 'upcoming' ──")
    historical_others_idx = full.index.tolist()
    last_10_match_ids = list(cmp_df.tail(10)["match_id"])

    per_match_vals = {}
    for mid in last_10_match_ids:
        match_row = full[full["match_id"] == mid].iloc[0].copy()
        side = "home" if match_row["home_team"] == TEAM else "away"
        match_row["result"] = np.nan
        match_row["home_goals"] = np.nan
        match_row["away_goals"] = np.nan
        hist = full[full["match_id"] != mid]
        per_call = pd.concat([hist, pd.DataFrame([match_row])], ignore_index=True)
        per_call["date"] = pd.to_datetime(per_call["date"], utc=True).dt.tz_localize(None)
        out = build_form_features(per_call)
        out_row = out[out["match_id"] == mid].iloc[0]
        col = f"{side}_w{W}_ppg"
        per_match_vals[mid] = float(out_row[col]) if pd.notna(out_row[col]) else np.nan

    cmp_df["per_match_w3_ppg"] = cmp_df["match_id"].map(per_match_vals)
    last_10_v2 = cmp_df.tail(10).copy()
    print(last_10_v2[["date", "side", "result", "manual_w3_ppg", "batch_w3_ppg",
                      "per_match_w3_ppg"]].to_string(index=False))

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n── Verdict ────────────────────────────────────────────────────")
    cmp = last_10_v2.dropna(subset=["manual_w3_ppg"])
    diff_batch = (cmp["batch_w3_ppg"] - cmp["manual_w3_ppg"]).abs()
    diff_per = (cmp["per_match_w3_ppg"] - cmp["manual_w3_ppg"]).abs()
    print(f"  max |batch − manual|     = {diff_batch.max():.6f}")
    print(f"  max |per_match − manual| = {diff_per.max():.6f}")
    if diff_batch.max() < 1e-6:
        print("  → BATCH agrees with manual ground truth.")
    else:
        print("  → BATCH DIVERGES from manual ground truth.")
    if diff_per.max() < 1e-6:
        print("  → PER-MATCH agrees with manual ground truth.")
    else:
        print("  → PER-MATCH DIVERGES from manual ground truth.")


if __name__ == "__main__":
    main()
