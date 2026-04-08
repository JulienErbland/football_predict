"use client";

import { useState, useMemo } from "react";
import { PredictionsData, Match } from "@/types/predictions";
import MatchCard from "./MatchCard";
import LeagueFilter from "./LeagueFilter";

interface PredictionsDashboardProps {
  data: PredictionsData;
}

export default function PredictionsDashboard({ data }: PredictionsDashboardProps) {
  const [selectedLeague, setSelectedLeague] = useState("All");

  const leagues = useMemo(
    () => [...new Set(data.matches.map((m) => m.league))],
    [data.matches]
  );

  const filtered = useMemo(
    () =>
      selectedLeague === "All"
        ? data.matches
        : data.matches.filter((m) => m.league === selectedLeague),
    [data.matches, selectedLeague]
  );

  const valueBetMatches = filtered.filter((m) => m.value_bets.length > 0);
  const regularMatches = filtered.filter((m) => m.value_bets.length === 0);

  return (
    <>
      <LeagueFilter
        leagues={leagues}
        selected={selectedLeague}
        onChange={setSelectedLeague}
      />

      {valueBetMatches.length > 0 && (
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-slate-800 mb-3 flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
            Value Bet Opportunities
            <span className="text-sm font-normal text-slate-400">
              ({valueBetMatches.length} match{valueBetMatches.length !== 1 ? "es" : ""})
            </span>
          </h2>
          <div className="flex flex-col gap-4">
            {valueBetMatches.map((match: Match) => (
              <MatchCard key={match.match_id} match={match} />
            ))}
          </div>
        </section>
      )}

      {regularMatches.length > 0 && (
        <section>
          {valueBetMatches.length > 0 && (
            <h2 className="text-lg font-semibold text-slate-800 mb-3">Other Matches</h2>
          )}
          <div className="flex flex-col gap-4">
            {regularMatches.map((match: Match) => (
              <MatchCard key={match.match_id} match={match} />
            ))}
          </div>
        </section>
      )}

      {filtered.length === 0 && (
        <p className="text-slate-400 text-center py-16">No upcoming matches found.</p>
      )}
    </>
  );
}
