"use client";

interface LeagueFilterProps {
  leagues: string[];
  selected: string;
  onChange: (league: string) => void;
}

export default function LeagueFilter({ leagues, selected, onChange }: LeagueFilterProps) {
  const all = ["All", ...leagues];

  return (
    <div className="flex flex-wrap gap-2 mb-6">
      {all.map((league) => (
        <button
          key={league}
          onClick={() => onChange(league)}
          className={`px-4 py-1.5 rounded-full text-sm font-medium border transition-colors ${
            selected === league
              ? "bg-slate-800 text-white border-slate-800"
              : "bg-white text-slate-600 border-slate-200 hover:border-slate-400"
          }`}
        >
          {league}
        </button>
      ))}
    </div>
  );
}
