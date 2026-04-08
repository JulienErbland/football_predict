"use client";

import { useState } from "react";
import { Match } from "@/types/predictions";
import ValueBetBadge from "./ValueBetBadge";

interface MatchCardProps {
  match: Match;
}

const confidenceColors: Record<string, string> = {
  high: "bg-emerald-100 text-emerald-800",
  medium: "bg-amber-100 text-amber-800",
  low: "bg-slate-100 text-slate-600",
};

const outcomeLabel: Record<string, string> = {
  home_win: "Home Win",
  draw: "Draw",
  away_win: "Away Win",
};

function ProbabilityBar({ homeWin, draw, awayWin }: { homeWin: number; draw: number; awayWin: number }) {
  return (
    <div className="mt-3">
      <div className="flex rounded-full overflow-hidden h-4" role="img" aria-label="Probability bar">
        <div
          className="bg-emerald-500 transition-all duration-700 ease-out"
          style={{ width: `${homeWin * 100}%` }}
          title={`Home win: ${(homeWin * 100).toFixed(0)}%`}
        />
        <div
          className="bg-slate-300 transition-all duration-700 ease-out"
          style={{ width: `${draw * 100}%` }}
          title={`Draw: ${(draw * 100).toFixed(0)}%`}
        />
        <div
          className="bg-blue-400 transition-all duration-700 ease-out"
          style={{ width: `${awayWin * 100}%` }}
          title={`Away win: ${(awayWin * 100).toFixed(0)}%`}
        />
      </div>
      <div className="flex justify-between text-xs text-slate-500 mt-1.5">
        <span>
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1" />
          Home {(homeWin * 100).toFixed(0)}%
        </span>
        <span>
          <span className="inline-block w-2 h-2 rounded-full bg-slate-300 mr-1" />
          Draw {(draw * 100).toFixed(0)}%
        </span>
        <span>
          <span className="inline-block w-2 h-2 rounded-full bg-blue-400 mr-1" />
          Away {(awayWin * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function EdgeCell({ value }: { value: number }) {
  const pct = (value * 100).toFixed(1);
  if (value >= 0.05) return <span className="font-semibold text-emerald-600">+{pct}%</span>;
  if (value >= 0) return <span className="text-slate-500">+{pct}%</span>;
  return <span className="text-red-400">{pct}%</span>;
}

export default function MatchCard({ match }: MatchCardProps) {
  const [expanded, setExpanded] = useState(false);
  const hasValueBets = match.value_bets.length > 0;
  const { prediction } = match;

  return (
    <div className={`bg-white rounded-xl border shadow-sm overflow-hidden ${
      hasValueBets ? "border-emerald-200" : "border-slate-200"
    }`}>
      {/* Value bet indicator bar */}
      {hasValueBets && (
        <div className="h-1 bg-emerald-400" />
      )}

      <div className="p-5">
        {/* Header: date + league */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
            {match.league}
          </span>
          <span className="text-xs text-slate-400">
            {new Date(match.date).toLocaleDateString("en-GB", {
              weekday: "short",
              month: "short",
              day: "numeric",
            })}
          </span>
        </div>

        {/* Teams */}
        <div className="flex items-center justify-between">
          <span className="font-semibold text-slate-800 text-lg">{match.home_team}</span>
          <span className="text-slate-400 text-sm font-medium">vs</span>
          <span className="font-semibold text-slate-800 text-lg text-right">{match.away_team}</span>
        </div>

        {/* Probability bar */}
        <ProbabilityBar
          homeWin={prediction.home_win}
          draw={prediction.draw}
          awayWin={prediction.away_win}
        />

        {/* Prediction badge */}
        <div className="mt-3 flex items-center gap-2">
          <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
            confidenceColors[prediction.confidence]
          }`}>
            {outcomeLabel[prediction.predicted_outcome]} — {
              prediction.confidence.charAt(0).toUpperCase() + prediction.confidence.slice(1)
            } confidence
          </span>
        </div>

        {/* Value bets */}
        {hasValueBets && (
          <div className="mt-3 flex flex-wrap gap-2">
            {match.value_bets.map((bet, i) => (
              <ValueBetBadge key={i} bet={bet} />
            ))}
          </div>
        )}

        {/* Odds comparison toggle */}
        {match.odds_comparison.length > 0 && (
          <button
            onClick={() => setExpanded((e) => !e)}
            className="mt-4 text-sm text-blue-600 hover:text-blue-800 font-medium transition-colors"
          >
            {expanded ? "Hide odds ▲" : "Show odds comparison ▼"}
          </button>
        )}

        {/* Odds table */}
        {expanded && match.odds_comparison.length > 0 && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="text-xs text-slate-500 uppercase border-b border-slate-100">
                  <th className="py-2 text-left font-medium">Bookmaker</th>
                  <th className="py-2 text-center font-medium">Home</th>
                  <th className="py-2 text-center font-medium">Draw</th>
                  <th className="py-2 text-center font-medium">Away</th>
                  <th className="py-2 text-center font-medium">H Edge</th>
                  <th className="py-2 text-center font-medium">D Edge</th>
                  <th className="py-2 text-center font-medium">A Edge</th>
                </tr>
              </thead>
              <tbody>
                {/* Model probabilities row */}
                <tr className="bg-slate-50 border-b border-slate-100">
                  <td className="py-2 font-semibold text-slate-700">Model</td>
                  <td className="py-2 text-center text-emerald-700 font-semibold">
                    {(prediction.home_win * 100).toFixed(0)}%
                  </td>
                  <td className="py-2 text-center text-slate-600 font-semibold">
                    {(prediction.draw * 100).toFixed(0)}%
                  </td>
                  <td className="py-2 text-center text-blue-700 font-semibold">
                    {(prediction.away_win * 100).toFixed(0)}%
                  </td>
                  <td colSpan={3} />
                </tr>
                {match.odds_comparison.map((odds, i) => {
                  const isValueRow = match.value_bets.some((vb) => vb.bookmaker === odds.bookmaker);
                  return (
                    <tr
                      key={i}
                      className={`border-b border-slate-50 ${isValueRow ? "bg-emerald-50/50" : ""}`}
                    >
                      <td className={`py-2 font-medium ${isValueRow ? "text-emerald-700 border-l-2 border-emerald-400 pl-2" : "text-slate-600"}`}>
                        {odds.bookmaker}
                      </td>
                      <td className="py-2 text-center text-slate-700">{odds.home_odds.toFixed(2)}</td>
                      <td className="py-2 text-center text-slate-700">{odds.draw_odds.toFixed(2)}</td>
                      <td className="py-2 text-center text-slate-700">{odds.away_odds.toFixed(2)}</td>
                      <td className="py-2 text-center"><EdgeCell value={odds.home_edge} /></td>
                      <td className="py-2 text-center"><EdgeCell value={odds.draw_edge} /></td>
                      <td className="py-2 text-center"><EdgeCell value={odds.away_edge} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
