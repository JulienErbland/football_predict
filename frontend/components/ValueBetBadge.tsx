import { ValueBet } from "@/types/predictions";

interface ValueBetBadgeProps {
  bet: ValueBet;
}

const outcomeLabel: Record<string, string> = {
  home_win: "Home Win",
  draw: "Draw",
  away_win: "Away Win",
};

export default function ValueBetBadge({ bet }: ValueBetBadgeProps) {
  return (
    <div className="inline-flex items-center gap-2 bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-1.5 text-sm">
      <span className="font-semibold text-emerald-800 capitalize">{bet.bookmaker}</span>
      <span className="text-slate-500">·</span>
      <span className="text-emerald-700">{outcomeLabel[bet.outcome]}</span>
      <span className="text-slate-500">·</span>
      <span className="font-bold text-emerald-600">+{(bet.edge * 100).toFixed(1)}% edge</span>
      <span className="text-slate-400 text-xs">Kelly {(bet.kelly * 100).toFixed(1)}%</span>
    </div>
  );
}
