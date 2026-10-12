"""Time-decayed Dixon-Coles models and full scoreline distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

_PROMOTED_RATING_CACHE: dict[tuple[int, int], list[tuple[float, float]]] = {}


@dataclass(frozen=True)
class ScorelineForecast:
    """A fixture's normalized scoreline probability matrix and goal means."""

    fixture_id: str
    home: str
    away: str
    matrix: np.ndarray
    xg_home: float
    xg_away: float


class DixonColesModel:
    """Weighted Poisson attack/defence model with the Dixon-Coles low-score term.

    Team attack and defence effects use a reference-team parameterization whose
    final effect is minus the sum of the others, enforcing sum-to-zero constraints.
    The training target is ``(1-alpha)*goals + alpha*npxG`` when xG is available.
    """

    def __init__(
        self,
        xi_candidates: Iterable[float] = (0.0, 0.001, 0.002, 0.003, 0.005),
        alpha_candidates: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        max_goals: int = 10,
    ) -> None:
        self.xi_candidates = tuple(xi_candidates)
        self.alpha_candidates = tuple(alpha_candidates)
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.mu = 0.0
        self.home_advantage = 0.0
        self.rho = 0.0
        self.xi = 0.0
        self.alpha = 0.0
        self.training_rows = 0
        self.training_max_date: pd.Timestamp | None = None
        self.promoted_priors: dict[str, tuple[float, float]] = {}
        self.promoted_average = (0.0, 0.0)
        self.promoted_teams: set[str] = set()

    @staticmethod
    def _dates(values: pd.Series) -> pd.Series:
        return pd.to_datetime(values, utc=True, errors="raise")

    def _strict_training_data(self, matches: pd.DataFrame, as_of_date: object) -> pd.DataFrame:
        required = {"kickoff_utc", "home", "away", "home_goals", "away_goals"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"Matches missing required columns: {sorted(missing)}")
        cutoff = pd.Timestamp(as_of_date)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        data = matches.copy()
        data["kickoff_utc"] = self._dates(data["kickoff_utc"])
        data = data.loc[data["kickoff_utc"] < cutoff].copy()
        data = data.dropna(subset=["home_goals", "away_goals", "home", "away"])
        data["home_goals"] = pd.to_numeric(data["home_goals"], errors="raise")
        data["away_goals"] = pd.to_numeric(data["away_goals"], errors="raise")
        if data.empty:
            raise ValueError("No completed matches strictly before as_of_date")
        return data.sort_values("kickoff_utc").reset_index(drop=True)

    def fit(self, matches: pd.DataFrame, as_of_date: object) -> "DixonColesModel":
        """Fit using only matches strictly earlier than the UTC prediction date."""
        data = self._strict_training_data(matches, as_of_date)
        self.training_rows = len(data)
        self.training_max_date = data["kickoff_utc"].max()
        self.teams = sorted(set(data["home"]) | set(data["away"]))
        if len(self.teams) < 2:
            raise ValueError("At least two teams are required to fit Dixon-Coles")

        # Select decay and xG blend on a chronological tail held out from fitting.
        split = max(1, int(len(data) * 0.8))
        tune = data.iloc[split:]
        prefix = data.iloc[:split]
        if len(tune) < 10 or len(prefix) < max(8, len(self.teams)):
            prefix, tune = data, data.iloc[0:0]
        candidates = self.alpha_candidates if self._has_npxg(data) else (0.0,)
        best = (float("inf"), self.xi_candidates[0], candidates[0])
        for xi in self.xi_candidates:
            for alpha in candidates:
                if tune.empty:
                    loss = self._fit_objective(data, xi, alpha, initial=None)[0]
                else:
                    fitted = self._fit_parameters(prefix, xi, alpha)
                    loss = self._mean_log_loss(tune, fitted, alpha)
                if np.isfinite(loss) and loss < best[0]:
                    best = (loss, xi, alpha)
        self.xi, self.alpha = best[1], best[2]
        params = self._fit_parameters(data, self.xi, self.alpha)
        self._assign_parameters(params)
        self._derive_promoted_priors(data)
        cutoff = pd.Timestamp(as_of_date)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        if self.training_max_date >= cutoff:
            raise AssertionError("Temporal cutoff violation in model fit")
        return self

    @staticmethod
    def _has_npxg(data: pd.DataFrame) -> bool:
        return {"npxg_home", "npxg_away"}.issubset(data.columns) and data[["npxg_home", "npxg_away"]].notna().all().all()

    def _targets(self, data: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        home = data["home_goals"].to_numpy(dtype=float)
        away = data["away_goals"].to_numpy(dtype=float)
        if alpha and self._has_npxg(data):
            home = (1 - alpha) * home + alpha * data["npxg_home"].to_numpy(dtype=float)
            away = (1 - alpha) * away + alpha * data["npxg_away"].to_numpy(dtype=float)
        return home, away

    def _unpack(self, params: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, float]:
        count = len(self.teams)
        mu, home_adv = params[:2]
        attack_free = params[2 : 2 + count - 1]
        defence_free = params[2 + count - 1 : 2 + 2 * (count - 1)]
        attack = np.r_[attack_free, -attack_free.sum()]
        defence = np.r_[defence_free, -defence_free.sum()]
        rho = params[-1]
        return mu, home_adv, attack, defence, rho

    def _rates(self, rows: pd.DataFrame, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, home_adv, attack, defence, _ = self._unpack(params)
        index = {team: i for i, team in enumerate(self.teams)}
        home_attack = rows["home"].map(index).fillna(-1).to_numpy(dtype=int)
        away_attack = rows["away"].map(index).fillna(-1).to_numpy(dtype=int)
        ha = np.array([attack[i] if i >= 0 else 0.0 for i in home_attack])
        hd = np.array([defence[i] if i >= 0 else 0.0 for i in home_attack])
        aa = np.array([attack[i] if i >= 0 else 0.0 for i in away_attack])
        ad = np.array([defence[i] if i >= 0 else 0.0 for i in away_attack])
        return np.exp(np.clip(mu + home_adv + ha - ad, -5, 4)), np.exp(np.clip(mu + aa - hd, -5, 4))

    @staticmethod
    def _tau(home_goals: np.ndarray, away_goals: np.ndarray, home_rate: np.ndarray, away_rate: np.ndarray, rho: float) -> np.ndarray:
        tau = np.ones(len(home_goals), dtype=float)
        tau[(home_goals == 0) & (away_goals == 0)] = 1 - home_rate[(home_goals == 0) & (away_goals == 0)] * away_rate[(home_goals == 0) & (away_goals == 0)] * rho
        tau[(home_goals == 0) & (away_goals == 1)] = 1 + home_rate[(home_goals == 0) & (away_goals == 1)] * rho
        tau[(home_goals == 1) & (away_goals == 0)] = 1 + away_rate[(home_goals == 1) & (away_goals == 0)] * rho
        tau[(home_goals == 1) & (away_goals == 1)] = 1 - rho
        return np.maximum(tau, 1e-9)

    def _fit_objective(self, data: pd.DataFrame, xi: float, alpha: float, initial: np.ndarray | None) -> tuple[float, np.ndarray]:
        params = self._fit_parameters(data, xi, alpha, initial)
        return self._negative_log_likelihood(params, data, xi, alpha), params

    def _negative_log_likelihood(self, params: np.ndarray, data: pd.DataFrame, xi: float, alpha: float) -> float:
        home_obs, away_obs = self._targets(data, alpha)
        home_rate, away_rate = self._rates(data, params)
        ll_home = home_obs * np.log(home_rate) - home_rate - gammaln(home_obs + 1)
        ll_away = away_obs * np.log(away_rate) - away_rate - gammaln(away_obs + 1)
        # The low-score correction is applied to integer observed goals, not fractional xG targets.
        _, _, _, _, actual_rho = self._unpack(params)
        actual_tau = self._tau(data["home_goals"].to_numpy(), data["away_goals"].to_numpy(), home_rate, away_rate, actual_rho)
        ll = ll_home + ll_away + np.log(actual_tau)
        age_days = (data["kickoff_utc"].max() - data["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
        weights = np.exp(-xi * age_days)
        return float(-np.sum(weights * ll) / np.sum(weights))

    def _fit_parameters(self, data: pd.DataFrame, xi: float, alpha: float, initial: np.ndarray | None = None) -> np.ndarray:
        self.teams = sorted(set(data["home"]) | set(data["away"]))
        count = len(self.teams)
        if initial is None:
            initial = np.zeros(2 + 2 * (count - 1) + 1)
            initial[0] = np.log(max(0.2, (data["home_goals"].mean() + data["away_goals"].mean()) / 2))
            initial[1] = 0.2
        bounds = [(None, None)] * (len(initial) - 1) + [(-0.2, 0.2)]
        result = minimize(
            self._negative_log_likelihood,
            initial,
            args=(data, xi, alpha),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"Dixon-Coles optimization failed: {result.message}")
        return result.x

    def _assign_parameters(self, params: np.ndarray) -> None:
        self.mu, self.home_advantage, attack, defence, self.rho = self._unpack(params)
        self.attack = dict(zip(self.teams, attack, strict=True))
        self.defence = dict(zip(self.teams, defence, strict=True))

    def _mean_log_loss(self, data: pd.DataFrame, params: np.ndarray, alpha: float) -> float:
        home_obs, away_obs = self._targets(data, alpha)
        home_rate, away_rate = self._rates(data, params)
        _, _, _, _, rho = self._unpack(params)
        home_score = home_obs * np.log(home_rate) - home_rate - gammaln(home_obs + 1)
        away_score = away_obs * np.log(away_rate) - away_rate - gammaln(away_obs + 1)
        tau = self._tau(data["home_goals"].to_numpy(), data["away_goals"].to_numpy(), home_rate, away_rate, rho)
        return float(-np.mean(home_score + away_score + np.log(tau)))

    def _derive_promoted_priors(self, data: pd.DataFrame) -> None:
        self.promoted_priors = {}
        self.promoted_teams = set()
        self.promoted_average = (0.0, 0.0)
        starts = data["kickoff_utc"].map(lambda value: value.year if value.month >= 7 else value.year - 1)
        data = data.assign(_season=starts)
        archive_start = int(data["_season"].min())
        previous_season_teams: set[str] = set()
        latest_season = None
        latest_season_teams: set[str] = set()
        historical_ratings: list[tuple[float, float]] = []
        for season, season_data in data.groupby("_season", sort=True):
            season_teams = set(season_data["home"]) | set(season_data["away"])
            if latest_season is None or int(season) > latest_season:
                previous_season_teams = latest_season_teams
                latest_season = int(season)
                latest_season_teams = season_teams
            if int(season) == archive_start or len(season_data) < 250:
                continue
            previous_teams = set(data.loc[data["_season"] == int(season) - 1, "home"]) | set(data.loc[data["_season"] == int(season) - 1, "away"])
            newcomers = sorted(season_teams - previous_teams)
            if not newcomers:
                continue
            cache_key = (int(season), len(season_data))
            if cache_key not in _PROMOTED_RATING_CACHE:
                season_model = DixonColesModel(xi_candidates=(0.0,), alpha_candidates=(0.0,), max_goals=self.max_goals)
                params = season_model._fit_parameters(season_data, xi=0.002, alpha=0.0)
                season_model._assign_parameters(params)
                _PROMOTED_RATING_CACHE[cache_key] = [
                    (season_model.attack.get(team, 0.0), season_model.defence.get(team, 0.0))
                    for team in newcomers
                ]
            historical_ratings.extend(_PROMOTED_RATING_CACHE[cache_key])
        if historical_ratings:
            self.promoted_average = tuple(np.mean(historical_ratings, axis=0))
        if latest_season is not None and latest_season_teams:
            self.promoted_teams = latest_season_teams - previous_season_teams

    def scoreline_matrix(self, home: str, away: str, max_goals: int | None = None) -> np.ndarray:
        """Return P(home goals, away goals), normalized on the requested grid."""
        if not self.attack:
            raise RuntimeError("Fit the model before predicting")
        home_prior = self.promoted_average if home in self.promoted_teams else (0.0, 0.0)
        away_prior = self.promoted_average if away in self.promoted_teams else (0.0, 0.0)
        home_att = home_prior[0] if home in self.promoted_teams else self.attack.get(home, 0.0)
        home_def = home_prior[1] if home in self.promoted_teams else self.defence.get(home, 0.0)
        away_att = away_prior[0] if away in self.promoted_teams else self.attack.get(away, 0.0)
        away_def = away_prior[1] if away in self.promoted_teams else self.defence.get(away, 0.0)
        home_rate = float(np.exp(self.mu + self.home_advantage + home_att - away_def))
        away_rate = float(np.exp(self.mu + away_att - home_def))
        goals = self.max_goals if max_goals is None else max_goals
        values = np.arange(goals + 1)
        hp = np.exp(values * np.log(home_rate) - home_rate - gammaln(values + 1))
        ap = np.exp(values * np.log(away_rate) - away_rate - gammaln(values + 1))
        matrix = np.outer(hp, ap)
        matrix[0, 0] *= 1 - home_rate * away_rate * self.rho
        if goals >= 1:
            matrix[0, 1] *= 1 + home_rate * self.rho
            matrix[1, 0] *= 1 + away_rate * self.rho
            matrix[1, 1] *= 1 - self.rho
        matrix = np.maximum(matrix, 0)
        return matrix / matrix.sum()

    def predict(self, fixtures: pd.DataFrame) -> list[ScorelineForecast]:
        """Produce normalized scoreline distributions for all supplied fixtures."""
        forecasts = []
        for row in fixtures.to_dict("records"):
            home, away = str(row["home"]), str(row["away"])
            matrix = self.scoreline_matrix(home, away)
            forecasts.append(
                ScorelineForecast(
                    fixture_id=str(row.get("fixture_id", f"{row.get('kickoff_utc')}:{home}:{away}")),
                    home=home,
                    away=away,
                    matrix=matrix,
                    xg_home=float(sum(i * matrix[i, :].sum() for i in range(matrix.shape[0]))),
                    xg_away=float(sum(j * matrix[:, j].sum() for j in range(matrix.shape[1]))),
                )
            )
        return forecasts
