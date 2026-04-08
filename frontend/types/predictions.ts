// TypeScript interfaces for predictions.json schema.
// These types are the single source of truth — use them throughout the app.

export interface OddsComparison {
  bookmaker: string;
  home_odds: number;
  draw_odds: number;
  away_odds: number;
  home_implied: number;
  draw_implied: number;
  away_implied: number;
  home_edge: number;
  draw_edge: number;
  away_edge: number;
}

export interface ValueBet {
  bookmaker: string;
  outcome: "home_win" | "draw" | "away_win";
  model_prob: number;
  bookmaker_odds: number;
  implied_prob: number;
  edge: number;
  kelly: number;
  confidence_tier: "high" | "medium" | "low";
}

export interface Prediction {
  home_win: number;
  draw: number;
  away_win: number;
  predicted_outcome: "home_win" | "draw" | "away_win";
  confidence: "high" | "medium" | "low";
}

export interface Match {
  match_id: number;
  league: string;
  date: string;
  home_team: string;
  away_team: string;
  prediction: Prediction;
  odds_comparison: OddsComparison[];
  value_bets: ValueBet[];
}

export interface PredictionsData {
  generated_at: string;
  matches: Match[];
}
