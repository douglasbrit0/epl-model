"""Time-decayed Dixon-Coles models and full scoreline distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from epl_model.seasons import season_labels, season_start_year

# Single-season newcomer ratings keyed by (season, rows, first/last kickoff, newcomers).
_PROMOTED_RATING_CACHE: dict[tuple, list[tuple[float, float]]] = {}


def dc_terms(eta_h_raw: np.ndarray, eta_a_raw: np.ndarray, rho: float, design: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-match Dixon-Coles log-likelihood and its derivatives w.r.t. both log-rates and rho.

    Log-rates are clipped to [-5, 4] (zero gradient outside). The low-score correction uses the
    integer observed goals, not fractional xG targets; floored tau terms get zero gradient.
    """
    eta_h, eta_a = np.clip(eta_h_raw, -5, 4), np.clip(eta_a_raw, -5, 4)
    lam, lam_a = np.exp(eta_h), np.exp(eta_a)
    s00, s01, s10, s11 = design["s00"], design["s01"], design["s10"], design["s11"]
    tau_raw = np.ones_like(lam)
    tau_raw[s00] = 1 - lam[s00] * lam_a[s00] * rho
    tau_raw[s01] = 1 + lam[s01] * rho
    tau_raw[s10] = 1 + lam_a[s10] * rho
    tau_raw[s11] = 1 - rho
    live = tau_raw > 1e-9
    tau = np.maximum(tau_raw, 1e-9)
    ll = design["home_obs"] * eta_h - lam + design["away_obs"] * eta_a - lam_a - design["const"] + np.log(tau)
    inv = np.where(live, 1 / tau, 0.0)
    g_h = design["home_obs"] - lam
    g_a = design["away_obs"] - lam_a
    g_h[s00] -= lam[s00] * lam_a[s00] * rho * inv[s00]
    g_a[s00] -= lam[s00] * lam_a[s00] * rho * inv[s00]
    g_h[s01] += lam[s01] * rho * inv[s01]
    g_a[s10] += lam_a[s10] * rho * inv[s10]
    g_rho = np.zeros_like(lam)
    g_rho[s00] = -lam[s00] * lam_a[s00] * inv[s00]
    g_rho[s01] = lam[s01] * inv[s01]
    g_rho[s10] = lam_a[s10] * inv[s10]
    g_rho[s11] = -inv[s11]
    g_h *= (eta_h_raw > -5) & (eta_h_raw < 4)
    g_a *= (eta_a_raw > -5) & (eta_a_raw < 4)
    return ll, g_h, g_a, g_rho


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

    Teams that did not play in the previous season (promoted sides) are shrunk toward an
    empirical-Bayes prior: the average first-season rating of past newcomers relative to the
    established teams, with precision equal to the inverse variance of those ratings. Teams
    with no data at all take that prior directly.
    """

    # A season needs this many matches before its newcomers inform the promoted-team prior.
    min_season_matches = 300

    def __init__(
        self,
        xi_candidates: Iterable[float] = (0.0, 0.001, 0.002, 0.003, 0.005),
        alpha_candidates: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        max_goals: int = 10,
        split_spells: bool = True,
        prior_scale: float = 1.0,
        home_shrink: float | None = None,
        effect_ridge: float = 0.0,
        effect_scale: float = 1.0,
        xg_columns: tuple[str, str] = ("npxg_home", "npxg_away"),
        covariates: tuple[str, ...] = (),
        covariate_ridge: float = 10.0,
    ) -> None:
        self.xi_candidates = tuple(xi_candidates)
        self.alpha_candidates = tuple(alpha_candidates)
        self.max_goals = max_goals
        self.split_spells = split_spells
        self.prior_scale = prior_scale
        # Experimental options (off by default): team-specific home advantage with a Gaussian
        # penalty of this precision; ridge penalty on all team effects; shrinking team effects
        # toward the established-team average at prediction time (a coherent recalibration).
        self.home_shrink = home_shrink
        self.effect_ridge = effect_ridge
        self.effect_scale = effect_scale
        self.xg_columns = xg_columns
        # Match-level adjustments to expected goals. For each name, column ``home_<name>`` is added to
        # the home team's log-rate and ``away_<name>`` to the away team's, with one shared fitted
        # coefficient (Gaussian penalty of precision covariate_ridge). Missing values count as 0.
        self.covariates = tuple(covariates)
        self.covariate_ridge = covariate_ridge
        self.covariate_coef: dict[str, float] = {}
        self.home_delta: dict[str, float] = {}
        self._centre = (0.0, 0.0)
        # Optional per-team fallback ratings for teams with no data (default: the single prior).
        self._unseen_by_team: dict[str, tuple[float, float]] = {}
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
        self.season: int | None = None
        self.prior_delta = (0.0, 0.0)
        self.prior_strength = 0.0
        self.promoted_average = (0.0, 0.0)
        self.promoted_teams: set[str] = set()
        self._unseen = (0.0, 0.0)

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
        cutoff = pd.Timestamp(as_of_date)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        self.season = season_start_year(cutoff)
        self._derive_promoted_priors(data, self.season)
        if self.split_spells:
            data = self._split_spells(data, self.season)

        # Select decay and xG blend on a chronological tail held out from fitting.
        split = max(1, int(len(data) * 0.8))
        tune = data.iloc[split:]
        prefix = data.iloc[:split]
        if len(tune) < 10 or len(prefix) < max(8, len(self.teams)):
            prefix, tune = data, data.iloc[0:0]
        candidates = self.alpha_candidates if self._has_npxg(data) else (0.0,)
        best = (float("inf"), self.xi_candidates[0], candidates[0])
        if len(self.xi_candidates) * len(candidates) == 1:
            tune = None  # a single fixed configuration needs no selection
        for xi in self.xi_candidates if tune is not None else ():
            for alpha in candidates:
                if tune.empty:
                    loss = self._fit_objective(data, xi, alpha, initial=None)[0]
                else:
                    fitted = self._fit_parameters(prefix, xi, alpha, season=season_start_year(tune["kickoff_utc"].iloc[0]))
                    loss = self._mean_log_loss(tune, fitted)
                if np.isfinite(loss) and loss < best[0]:
                    best = (loss, xi, alpha)
        self.xi, self.alpha = best[1], best[2]
        params = self._fit_parameters(data, self.xi, self.alpha, season=self.season)
        self._assign_parameters(params)
        labels = season_labels(data["kickoff_utc"])
        current = data.loc[labels == self.season]
        previous = data.loc[labels == self.season - 1]
        if not previous.empty:
            established = set(previous["home"]) | set(previous["away"])
            self.promoted_teams = (set(current["home"]) | set(current["away"])) - established
        if self.training_max_date >= cutoff:
            raise AssertionError("Temporal cutoff violation in model fit")
        return self

    def _has_npxg(self, data: pd.DataFrame) -> bool:
        """True when the xG columns exist and at least some training matches have xG."""
        columns = list(self.xg_columns)
        return set(columns).issubset(data.columns) and bool(data[columns].notna().all(axis=1).any())

    def _targets(self, data: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Blend goals with xG where xG exists; matches without xG keep their goals.

        xG is rescaled so its total matches goals over the same training matches, so blending
        changes the information used, not the overall scoring level.
        """
        home = data["home_goals"].to_numpy(dtype=float)
        away = data["away_goals"].to_numpy(dtype=float)
        if alpha and self._has_npxg(data):
            xg_home = data[self.xg_columns[0]].to_numpy(dtype=float)
            xg_away = data[self.xg_columns[1]].to_numpy(dtype=float)
            known = np.isfinite(xg_home) & np.isfinite(xg_away)
            scale = (home[known].sum() + away[known].sum()) / max(xg_home[known].sum() + xg_away[known].sum(), 1e-9)
            home = np.where(known, (1 - alpha) * home + alpha * scale * np.where(known, xg_home, 0.0), home)
            away = np.where(known, (1 - alpha) * away + alpha * scale * np.where(known, xg_away, 0.0), away)
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

    def _home_deltas(self, params: np.ndarray) -> np.ndarray:
        count = len(self.teams)
        if self.home_shrink is None:
            return np.zeros(count)
        start = 2 + 2 * (count - 1)
        return params[start : start + count]

    def _coef_start(self) -> int:
        count = len(self.teams)
        return 2 + 2 * (count - 1) + (count if self.home_shrink is not None else 0)

    def _coefs(self, params: np.ndarray) -> np.ndarray:
        start = self._coef_start()
        return params[start : start + len(self.covariates)]

    def _covariate_matrices(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.covariates:
            return np.zeros((len(rows), 0)), np.zeros((len(rows), 0))
        def matrix(side: str) -> np.ndarray:
            columns = [rows[f"{side}_{name}"] if f"{side}_{name}" in rows else pd.Series(0.0, index=rows.index) for name in self.covariates]
            return np.nan_to_num(np.column_stack([pd.to_numeric(c, errors="coerce").to_numpy(float) for c in columns]))
        return matrix("home"), matrix("away")

    def _team_indices(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        index = {team: i for i, team in enumerate(self.teams)}
        home = rows["home"].map(index).fillna(-1).to_numpy(dtype=int)
        away = rows["away"].map(index).fillna(-1).to_numpy(dtype=int)
        return home, away

    def _rates(self, rows: pd.DataFrame, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, home_adv, attack, defence, _ = self._unpack(params)
        home, away = self._team_indices(rows)
        # Unknown teams (index -1) take the promoted-team prior.
        attack, defence = np.r_[attack, self._unseen[0]], np.r_[defence, self._unseen[1]]
        deltas = np.r_[self._home_deltas(params), 0.0]
        x_home, x_away = self._covariate_matrices(rows)
        coefs = self._coefs(params)
        return (
            np.exp(np.clip(mu + home_adv + deltas[home] + attack[home] - defence[away] + x_home @ coefs, -5, 4)),
            np.exp(np.clip(mu + attack[away] - defence[home] + x_away @ coefs, -5, 4)),
        )

    @staticmethod
    def _tau(home_goals: np.ndarray, away_goals: np.ndarray, home_rate: np.ndarray, away_rate: np.ndarray, rho: float) -> np.ndarray:
        tau = np.ones(len(home_goals), dtype=float)
        tau[(home_goals == 0) & (away_goals == 0)] = 1 - home_rate[(home_goals == 0) & (away_goals == 0)] * away_rate[(home_goals == 0) & (away_goals == 0)] * rho
        tau[(home_goals == 0) & (away_goals == 1)] = 1 + home_rate[(home_goals == 0) & (away_goals == 1)] * rho
        tau[(home_goals == 1) & (away_goals == 0)] = 1 + away_rate[(home_goals == 1) & (away_goals == 0)] * rho
        tau[(home_goals == 1) & (away_goals == 1)] = 1 - rho
        return np.maximum(tau, 1e-9)

    def _fit_objective(self, data: pd.DataFrame, xi: float, alpha: float, initial: np.ndarray | None) -> tuple[float, np.ndarray]:
        params = self._fit_parameters(data, xi, alpha, initial, season=self.season)
        return self._negative_log_likelihood(params, data, xi, alpha), params

    @staticmethod
    def _split_spells(data: pd.DataFrame, season: int) -> pd.DataFrame:
        """Rename matches from a team's earlier EPL spells so they do not set its current rating.

        A team's current spell is its unbroken run of seasons up to ``season`` (or ``season - 1``).
        Earlier matches, and all matches of teams absent from both seasons, become a separate
        entity; a returning team then starts from the promoted-team prior.
        """
        labels = season_labels(data["kickoff_utc"])
        played = {}
        for column in ("home", "away"):
            for team, group in labels.groupby(data[column]):
                played.setdefault(team, set()).update(int(value) for value in group.unique())
        spell_start = {}
        for team, seasons in played.items():
            latest = season if season in seasons else season - 1
            if latest not in seasons:
                spell_start[team] = season + 1  # not currently in the league: every match is old
                continue
            while latest - 1 in seasons:
                latest -= 1
            spell_start[team] = latest
        data = data.copy()
        for column in ("home", "away"):
            old = labels < data[column].map(spell_start)
            data.loc[old, column] = data.loc[old, column] + " (earlier spell)"
        return data

    def _promotion_penalty(self, data: pd.DataFrame, season: int | None) -> dict | None:
        """Shrink teams absent from the previous season toward the newcomer prior."""
        if season is None:
            return None
        labels = season_labels(data["kickoff_utc"])
        previous = data.loc[labels == season - 1]
        if previous.empty:
            return None
        established = set(previous["home"]) | set(previous["away"])
        index = {team: i for i, team in enumerate(self.teams)}
        return {
            "established": np.array(sorted(index[t] for t in established), dtype=int),
            "newcomers": np.array(sorted(i for t, i in index.items() if t not in established), dtype=int),
            "delta": self.prior_delta,
            "strength": self.prior_strength,
        }

    def _design(self, data: pd.DataFrame, xi: float, alpha: float, penalty: dict | None = None) -> dict[str, np.ndarray]:
        """Precompute everything the likelihood needs that does not depend on the parameters."""
        home_obs, away_obs = self._targets(data, alpha)
        home_goals = data["home_goals"].to_numpy()
        away_goals = data["away_goals"].to_numpy()
        age_days = (data["kickoff_utc"].max() - data["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
        weights = np.exp(-xi * age_days)
        home, away = self._team_indices(data)
        return {
            "home_obs": home_obs,
            "away_obs": away_obs,
            "const": gammaln(home_obs + 1) + gammaln(away_obs + 1),
            "s00": (home_goals == 0) & (away_goals == 0),
            "s01": (home_goals == 0) & (away_goals == 1),
            "s10": (home_goals == 1) & (away_goals == 0),
            "s11": (home_goals == 1) & (away_goals == 1),
            "w": weights / weights.sum(),
            "weight_sum": weights.sum(),
            "home": home,
            "away": away,
            "penalty": penalty,
            "covariates": self._covariate_matrices(data),
        }

    def _nll_and_grad(self, params: np.ndarray, design: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
        """Weighted mean negative log-likelihood and its exact gradient."""
        count = len(self.teams)
        mu, home_adv, attack, defence, rho = self._unpack(params)
        home, away, w = design["home"], design["away"], design["w"]
        deltas = self._home_deltas(params)
        x_home, x_away = design["covariates"]
        coefs = self._coefs(params)
        eta_h_raw = mu + home_adv + deltas[home] + attack[home] - defence[away] + x_home @ coefs
        eta_a_raw = mu + attack[away] - defence[home] + x_away @ coefs
        ll, g_h, g_a, g_rho = dc_terms(eta_h_raw, eta_a_raw, rho, design)
        wg_h, wg_a = w * g_h, w * g_a

        g_attack = np.bincount(home, wg_h, count) + np.bincount(away, wg_a, count)
        g_defence = -np.bincount(away, wg_h, count) - np.bincount(home, wg_a, count)
        grad = np.empty_like(params)
        grad[0] = wg_h.sum() + wg_a.sum()
        grad[1] = wg_h.sum()
        # The last team's effect is minus the sum of the free effects.
        grad[2 : 2 + count - 1] = g_attack[:-1] - g_attack[-1]
        grad[2 + count - 1 : 2 + 2 * (count - 1)] = g_defence[:-1] - g_defence[-1]
        grad[-1] = np.sum(w * g_rho)
        if self.home_shrink is not None:
            start = 2 + 2 * (count - 1)
            grad[start : start + count] = np.bincount(home, wg_h, count)
        if self.covariates:
            first = self._coef_start()
            grad[first : first + len(self.covariates)] = x_home.T @ wg_h + x_away.T @ wg_a
        value, grad = float(-np.sum(w * ll)), -grad

        if self.covariates:
            scale = self.covariate_ridge / design["weight_sum"]
            value += 0.5 * scale * float(coefs @ coefs)
            grad[first : first + len(self.covariates)] += scale * coefs

        if self.home_shrink is not None:
            scale = self.home_shrink / design["weight_sum"]
            value += 0.5 * scale * float(deltas @ deltas)
            grad[start : start + count] += scale * deltas
        if self.effect_ridge > 0:
            scale = self.effect_ridge / design["weight_sum"]
            for offset, effects in ((2, attack), (2 + count - 1, defence)):
                value += 0.5 * scale * float(effects @ effects)
                grad[offset : offset + count - 1] += scale * (effects[:-1] - effects[-1])

        penalty = design["penalty"]
        if penalty is not None and np.any(np.asarray(penalty["strength"]) > 0) and len(penalty["newcomers"]):
            # Gaussian prior on newcomers, in total-weight units so recent matches count ~1 each.
            scale = penalty["strength"] / design["weight_sum"]
            new, est = penalty["newcomers"], penalty["established"]
            for offset, effects, delta in ((2, attack, penalty["delta"][0]), (2 + count - 1, defence, penalty["delta"][1])):
                # delta and strength may be scalars or per-newcomer arrays (e.g. separate priors
                # for clubs arriving from the league above and below).
                residual = effects[new] - effects[est].mean() - delta
                value += 0.5 * float(np.sum(scale * residual * residual))
                g_effects = np.zeros(count)
                g_effects[new] += scale * residual
                g_effects[est] -= np.sum(scale * residual) / len(est)
                grad[offset : offset + count - 1] += g_effects[:-1] - g_effects[-1]
        return value, grad

    def _negative_log_likelihood(self, params: np.ndarray, data: pd.DataFrame, xi: float, alpha: float) -> float:
        return self._nll_and_grad(params, self._design(data, xi, alpha))[0]

    def _fit_parameters(self, data: pd.DataFrame, xi: float, alpha: float, initial: np.ndarray | None = None, season: int | None = None) -> np.ndarray:
        self.teams = sorted(set(data["home"]) | set(data["away"]))
        penalty = self._promotion_penalty(data, season)
        count = len(self.teams)
        if initial is None:
            initial = np.zeros(2 + 2 * (count - 1) + (count if self.home_shrink is not None else 0) + len(self.covariates) + 1)
            initial[0] = np.log(max(0.2, (data["home_goals"].mean() + data["away_goals"].mean()) / 2))
            initial[1] = 0.2
        bounds = [(None, None)] * (len(initial) - 1) + [(-0.2, 0.2)]
        result = minimize(
            self._nll_and_grad,
            initial,
            args=(self._design(data, xi, alpha, penalty),),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"Dixon-Coles optimization failed: {result.message}")
        _, _, attack, defence, _ = self._unpack(result.x)
        if penalty is not None:
            established = penalty["established"]
            self._centre = (float(attack[established].mean()), float(defence[established].mean()))
            self._unseen = (
                float(attack[established].mean() + self.prior_delta[0]),
                float(defence[established].mean() + self.prior_delta[1]),
            )
        else:
            self._centre = (float(attack.mean()), float(defence.mean()))
            self._unseen = self._centre
        self.promoted_average = self._unseen
        return result.x

    def _assign_parameters(self, params: np.ndarray) -> None:
        self.mu, self.home_advantage, attack, defence, self.rho = self._unpack(params)
        self.attack = dict(zip(self.teams, attack, strict=True))
        self.defence = dict(zip(self.teams, defence, strict=True))
        self.home_delta = dict(zip(self.teams, self._home_deltas(params), strict=True))
        self.covariate_coef = dict(zip(self.covariates, (float(c) for c in self._coefs(params)), strict=True))

    def _mean_log_loss(self, data: pd.DataFrame, params: np.ndarray) -> float:
        """Scoreline log-loss on actual goals, so candidates with different xG blends are comparable."""
        home_obs, away_obs = self._targets(data, 0.0)
        home_rate, away_rate = self._rates(data, params)
        _, _, _, _, rho = self._unpack(params)
        home_score = home_obs * np.log(home_rate) - home_rate - gammaln(home_obs + 1)
        away_score = away_obs * np.log(away_rate) - away_rate - gammaln(away_obs + 1)
        tau = self._tau(data["home_goals"].to_numpy(), data["away_goals"].to_numpy(), home_rate, away_rate, rho)
        return float(-np.mean(home_score + away_score + np.log(tau)))

    def _derive_promoted_priors(self, data: pd.DataFrame, season: int) -> None:
        """Estimate the newcomer prior from completed seasons strictly before ``season``."""
        self.prior_delta, self.prior_strength = (0.0, 0.0), 0.0
        labels = season_labels(data["kickoff_utc"])
        ratings: list[tuple[float, float]] = []
        for past in sorted(set(labels)):
            past = int(past)
            season_data = data.loc[labels == past]
            previous = data.loc[labels == past - 1]
            if past >= season or previous.empty or len(season_data) < self.min_season_matches:
                continue
            newcomers = sorted((set(season_data["home"]) | set(season_data["away"])) - (set(previous["home"]) | set(previous["away"])))
            if not newcomers:
                continue
            key = (past, len(season_data), season_data["kickoff_utc"].min(), season_data["kickoff_utc"].max(), tuple(newcomers))
            if key not in _PROMOTED_RATING_CACHE:
                # Single-season fit: effects are relative to that season's average team.
                season_model = DixonColesModel(xi_candidates=(0.0,), alpha_candidates=(0.0,), max_goals=self.max_goals)
                season_model._assign_parameters(season_model._fit_parameters(season_data, xi=0.0, alpha=0.0))
                _PROMOTED_RATING_CACHE[key] = [(season_model.attack[team], season_model.defence[team]) for team in newcomers]
            ratings.extend(_PROMOTED_RATING_CACHE[key])
        if ratings:
            values = np.array(ratings)
            self.prior_delta = (float(values[:, 0].mean()), float(values[:, 1].mean()))
            variance = float(np.mean(np.square(values - values.mean(axis=0))))
            self.prior_strength = self.prior_scale / max(variance, 0.01)

    def scoreline_matrix(self, home: str, away: str, max_goals: int | None = None, offsets: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Return P(home goals, away goals), normalized on the requested grid.

        ``offsets`` are added to the home and away log-rates (fitted covariate adjustments).
        """
        if not self.attack:
            raise RuntimeError("Fit the model before predicting")
        # Promoted teams use their shrunk fitted effects; teams never seen use the prior.
        home_unseen = self._unseen_by_team.get(home, self._unseen)
        away_unseen = self._unseen_by_team.get(away, self._unseen)
        home_att = self.attack.get(home, home_unseen[0])
        home_def = self.defence.get(home, home_unseen[1])
        away_att = self.attack.get(away, away_unseen[0])
        away_def = self.defence.get(away, away_unseen[1])
        if self.effect_scale != 1.0:
            k, (centre_att, centre_def) = self.effect_scale, self._centre
            home_att, away_att = centre_att + k * (home_att - centre_att), centre_att + k * (away_att - centre_att)
            home_def, away_def = centre_def + k * (home_def - centre_def), centre_def + k * (away_def - centre_def)
        home_att += self.home_delta.get(home, 0.0)
        # Same log-rate bounds as the likelihood, so sparse early fits cannot produce zero rates.
        home_rate = float(np.exp(np.clip(self.mu + self.home_advantage + home_att - away_def + offsets[0], -5, 4)))
        away_rate = float(np.exp(np.clip(self.mu + away_att - home_def + offsets[1], -5, 4)))
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
        coefs = np.array([self.covariate_coef.get(name, 0.0) for name in self.covariates])
        x_home, x_away = self._covariate_matrices(fixtures.reset_index(drop=True))
        for position, row in enumerate(fixtures.to_dict("records")):
            home, away = str(row["home"]), str(row["away"])
            offsets = (float(x_home[position] @ coefs), float(x_away[position] @ coefs)) if self.covariates else (0.0, 0.0)
            matrix = self.scoreline_matrix(home, away, offsets=offsets)
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
