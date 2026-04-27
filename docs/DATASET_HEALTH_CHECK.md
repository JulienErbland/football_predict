# Dataset Health Check

Source: `backend/data/processed/all_matches.parquet`
Generated: 2026-04-26T12:22:18

Status legend: ✅ pass · ⚠️ check · ❌ fail

## 1. Basic shape

- **Total rows**: 7156
- **Date range**: 2021-08-06 → 2025-05-25
- **Leagues**: ['BL1', 'FL1', 'PD', 'PL', 'SA']
- **Seasons**: [2021, 2022, 2023, 2024]
- **Columns**: 13 → ['match_id', 'league', 'season', 'matchday', 'date', 'home_team_id', 'home_team', 'away_team_id', 'away_team', 'home_goals', 'away_goals', 'result', 'referee']

## 2. Completeness (completed matches only)

```
home_team     0
away_team     0
home_goals    0
away_goals    0
date          0
league        0
```
✅ All core columns fully populated for completed matches.

## 3. Team-name consistency

- **Total unique team names**: 125
- **Sample (first 12)**: ['Ajaccio', 'Alaves', 'Almeria', 'Angers', 'Arsenal', 'Aston Villa', 'Atalanta', 'Ath Bilbao', 'Ath Madrid', 'Augsburg', 'Auxerre', 'Barcelona']
✅ No obvious duplicate team-name variants.

## 4. Duplicate matches

- **Duplicate (date, home, away) rows**: 0
✅ No duplicate matches.

## 5. Chronological order on disk

- **`date` monotonic non-decreasing**: False
ℹ️ Data isn't pre-sorted on disk. Feature builders sort internally — this is fine but worth noting.

## 6. Result column matches goal counts

- **Mismatched result/goals rows**: 0
✅ All `result` values match the score.

## 7. Matches per league

```
league
BL1    1224
FL1    1372
PD     1520
PL     1520
SA     1520
```

## 8. Per-(league, season) row counts

```
league  season
BL1     2021      306
        2022      306
        2023      306
        2024      306
FL1     2021      380
        2022      380
        2023      306
        2024      306
PD      2021      380
        2022      380
        2023      380
        2024      380
PL      2021      380
        2022      380
        2023      380
        2024      380
SA      2021      380
        2022      380
        2023      380
        2024      380
```
✅ Per-(league, season) counts look plausible.

## 9. Goals distribution

```
        home_goals   away_goals
count  7156.000000  7156.000000
mean      1.549329     1.272079
std       1.310474     1.173148
min       0.000000     0.000000
25%       1.000000     0.000000
50%       1.000000     1.000000
75%       2.000000     2.000000
max       9.000000     8.000000
```

## 10. Result distribution (completed)

```
result
H    0.434181
A    0.312186
D    0.253633
```

## 11. Home advantage

- **Home-win rate**: 0.434
✅ Home-win rate inside expected band [0.40, 0.50].

## 12. Team match counts (top-5 vs bottom-5 by 'home' appearances)

- **Top-5**: {'Brentford': 76, 'Man United': 76, 'Chelsea': 76, 'Everton': 76, 'Newcastle': 76}
- **Bottom-5**: {'Greuther Furth': 17, 'Schalke 04': 17, 'Darmstadt': 17, 'St Pauli': 17, 'Holstein Kiel': 17}
✅ Team home-count spread is reasonable.

## 13. xG availability

ℹ️ No `home_xg` column in this parquet (xG is built lazily).

## 14. Date gaps > 60 days

- **# gaps > 60 days**: 3
```
      date league  season  gap_days
2022-08-05     PL    2022      75.0
2023-08-11     PD    2023      68.0
2024-08-15     PD    2024      74.0
```
ℹ️ Long gaps are expected at season boundaries — verify the dates.

## 15. Upcoming-team consistency

ℹ️ No `upcoming_matches.parquet` cached on disk; upcoming fixtures are fetched from football-data.org at predict time. T0.1 introduced a separate name-normalisation regression test that covers this surface (see `backend/tests/`).

## 16. Arsenal — last 10 matches (raw)

```
      date   home_team      away_team  home_goals  away_goals  result result_label
2025-03-16     Arsenal        Chelsea           1           0       0            H
2025-04-01     Arsenal         Fulham           2           1       0            H
2025-04-05     Everton        Arsenal           1           1       1            D
2025-04-12     Arsenal      Brentford           1           1       1            D
2025-04-20     Ipswich        Arsenal           0           4       2            A
2025-04-23     Arsenal Crystal Palace           2           2       1            D
2025-05-03     Arsenal    Bournemouth           1           2       2            A
2025-05-11   Liverpool        Arsenal           2           2       1            D
2025-05-18     Arsenal      Newcastle           1           0       0            H
2025-05-25 Southampton        Arsenal           1           2       2            A
```

## 17. Arsenal — manual rolling PPG (window=3, 5, 10)

```
      date side result  arsenal_points   ppg_w3  ppg_w5  ppg_w10
2025-03-16 home      H             3.0 0.666667     1.6      1.9
2025-04-01 home      H             3.0 1.666667     1.6      1.9
2025-04-05 away      D             1.0 2.333333     1.6      2.1
2025-04-12 home      D             1.0 2.333333     1.8      1.9
2025-04-20 away      A             3.0 1.666667     1.8      1.9
2025-04-23 home      D             1.0 1.666667     2.2      1.9
2025-05-03 home      A             0.0 1.666667     1.8      1.7
2025-05-11 away      D             1.0 1.333333     1.2      1.4
2025-05-18 home      H             3.0 0.666667     1.2      1.5
2025-05-25 away      A             3.0 1.333333     1.6      1.7
```
Convention: `ppg_wN` at row *t* uses Arsenal's previous N matches (strictly before *t*). This is the ground truth the form features should match.

## 18. form.py BATCH vs PER-MATCH vs ground truth (Arsenal w=3 PPG)

```
  match_id       date  manual_w3_ppg  batch_w3_ppg  per_match_w3_ppg
   6272209 2025-03-16       0.666667      0.666667          0.666667
3978114884 2025-04-01       1.666667      1.666667          1.666667
1389627574 2025-04-05       2.333333      2.333333          2.333333
3210950483 2025-04-12       2.333333      2.333333          2.333333
 190130257 2025-04-20       1.666667      1.666667          1.666667
2391187562 2025-04-23       1.666667      1.666667          1.666667
3750883464 2025-05-03       1.666667      1.666667          1.666667
    988845 2025-05-11       1.333333      1.333333          1.333333
3899057240 2025-05-18       0.666667      0.666667          0.666667
 985207538 2025-05-25       1.333333      1.333333          1.333333
```
- max |batch − manual|     = **0.000000**
- max |per_match − manual| = **0.000000**
✅ BATCH path matches manual ground truth.
✅ PER-MATCH path matches manual ground truth.

## Summary

✅ **Dataset structure is clean.** No nulls, dupes, broken team names, mis-coded results, or unusual distributions.
