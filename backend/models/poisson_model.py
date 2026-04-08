"""
Dixon-Coles Poisson regression model for football match prediction.

The standard independent Poisson model assumes home goals ~ Pois(lambda_h)
and away goals ~ Pois(lambda_a) independently. Dixon and Coles (1997) showed
that low-scoring combinations (0-0, 1-0, 0-1, 1-1) are systematically
over/under-represented. They add a correction factor rho (ρ) to the likelihood
that adjusts joint probabilities for these four scorelines.

Attack/defence strengths are estimated via maximum likelihood on historical data.
For prediction, we compute a 10x10 scoreline probability grid, then sum:
    P(home win) = sum of P(h>a)
    P(draw)     = sum of P(h==a)
    P(away win) = sum of P(h<a)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from models.base import BaseModel


class PoissonModel(BaseModel):
    """
    Dixon-Coles Poisson model estimating team attack/defence strengths.

    Parameters fit via MLE on historical match data.
    The model is especially useful as a baseline when ML training data is limited,
    because it has a principled statistical interpretation and doesn't overfit.
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self._attack: dict[str, float] = {}
        self._defence: dict[str, float] = {}
        self._home_advantage: float = 0.0
        self._rho: float = 0.0  # Dixon-Coles correction strength
        self._teams: list[str] = []

    def _dixon_coles_correction(self, home_g: int, away_g: int,
                                 lambda_h: float, lambda_a: float) -> float:
        """
        Dixon-Coles correction factor tau for low-scoring matches.

        Adjusts the joint probability of the four corner scorelines:
        (0,0), (1,0), (0,1), (1,1) to fix the independence assumption.
        For all other scorelines, correction = 1 (no adjustment).
        """
        rho = self._rho
        if home_g == 0 and away_g == 0:
            return 1 - lambda_h * lambda_a * rho
        elif home_g == 1 and away_g == 0:
            return 1 + lambda_a * rho
        elif home_g == 0 and away_g == 1:
            return 1 + lambda_h * rho
        elif home_g == 1 and away_g == 1:
            return 1 - rho
        return 1.0

    def _score_prob(self, home_team: str, away_team: str) -> np.ndarray:
        """
        Compute the (max_goals+1) x (max_goals+1) scoreline probability matrix.

        Rows = home goals, columns = away goals.
        """
        atk_h = self._attack.get(home_team, 1.0)
        def_h = self._defence.get(home_team, 1.0)
        atk_a = self._attack.get(away_team, 1.0)
        def_a = self._defence.get(away_team, 1.0)
        lambda_h = atk_h * def_a * self._home_advantage
        lambda_a = atk_a * def_h
        n = self.max_goals + 1
        matrix = np.zeros((n, n))
        for h in range(n):
            for a in range(n):
                p_h = poisson.pmf(h, lambda_h)
                p_a = poisson.pmf(a, lambda_a)
                tau = self._dixon_coles_correction(h, a, lambda_h, lambda_a)
                matrix[h, a] = p_h * p_a * tau
        # Renormalise because correction breaks the sum-to-1 property
        matrix /= matrix.sum()
        return matrix

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit attack/defence strengths from match data passed via kwargs.

        Expects kwargs to contain: home_teams (list), away_teams (list),
        home_goals (list), away_goals (list).

        X/y are ignored (Poisson model doesn't use feature vectors);
        they're included only for API consistency with BaseModel.
        """
        home_teams = kwargs.get("home_teams", [])
        away_teams = kwargs.get("away_teams", [])
        home_goals = kwargs.get("home_goals", [])
        away_goals = kwargs.get("away_goals", [])

        self._teams = sorted(set(home_teams) | set(away_teams))
        n_teams = len(self._teams)
        team_idx = {t: i for i, t in enumerate(self._teams)}

        # Parameter vector: [attack_0..n, defence_0..n, home_adv, rho]
        x0 = np.ones(n_teams * 2 + 2)
        x0[n_teams * 2] = 1.1  # Slight home advantage
        x0[n_teams * 2 + 1] = -0.1  # Slight negative rho (typical value)

        def neg_log_likelihood(params):
            attack = params[:n_teams]
            defence = params[n_teams:n_teams * 2]
            home_adv = params[n_teams * 2]
            rho = params[n_teams * 2 + 1]
            ll = 0.0
            for h_team, a_team, h_g, a_g in zip(
                home_teams, away_teams, home_goals, away_goals
            ):
                hi = team_idx[h_team]
                ai = team_idx[a_team]
                lam_h = max(attack[hi] * defence[ai] * home_adv, 1e-6)
                lam_a = max(attack[ai] * defence[hi], 1e-6)
                tau = _dc_tau(int(h_g), int(a_g), lam_h, lam_a, rho)
                ll += (
                    poisson.logpmf(int(h_g), lam_h)
                    + poisson.logpmf(int(a_g), lam_a)
                    + np.log(max(tau, 1e-10))
                )
            return -ll

        result = minimize(
            neg_log_likelihood, x0, method="L-BFGS-B",
            bounds=[(0.01, None)] * (n_teams * 2) + [(0.5, 3.0), (-1.0, 1.0)],
            options={"maxiter": 500},
        )
        params = result.x
        self._attack = {t: params[i] for i, t in enumerate(self._teams)}
        self._defence = {t: params[n_teams + i] for i, t in enumerate(self._teams)}
        self._home_advantage = params[n_teams * 2]
        self._rho = params[n_teams * 2 + 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Not used directly — Poisson model needs team names, not feature vectors.

        For API compatibility, expects X to be an object array of (home_team, away_team)
        tuples. In practice, predict_from_teams is more convenient.
        """
        return np.array([
            self.predict_from_teams(row[0], row[1]) for row in X
        ])

    def predict_from_teams(self, home_team: str, away_team: str) -> np.ndarray:
        """Return [p_home_win, p_draw, p_away_win] for a specific matchup."""
        matrix = self._score_prob(home_team, away_team)
        p_home = float(np.tril(matrix, -1).sum())  # home_goals > away_goals
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())
        return np.array([p_home, p_draw, p_away])


def _dc_tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Scalar Dixon-Coles tau for use in MLE (outside the class for speed)."""
    if h == 0 and a == 0:
        return 1 - lam_h * lam_a * rho
    elif h == 1 and a == 0:
        return 1 + lam_a * rho
    elif h == 0 and a == 1:
        return 1 + lam_h * rho
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0
