"""Dynamic Dixon-Coles: team strengths follow a Gaussian random walk over fixed time blocks.

This is the MAP estimate of a state-space model. Instead of down-weighting old matches with
exponential decay, every match counts fully and each team's attack and defence may move between
consecutive blocks with a Gaussian step. The league scoring level and home advantage also follow
(stiffer) random walks, so changes such as matches without crowds are absorbed. Steps across a
season boundary can have a larger variance and optional mean reversion. Newcomers, earlier spells
and unseen teams are handled exactly as in the static model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from epl_model.models.dixon_coles import DixonColesModel, dc_terms
from epl_model.seasons import season_labels, season_start_year


class DynamicDixonColesModel(DixonColesModel):
    """Random-walk team effects; predictions use the effects of the block containing the cutoff."""

    def __init__(
        self,
        block_days: int = 28,
        step_precision: float = 300.0,
        summer_variance: float = 4.0,
        summer_reversion: float = 1.0,
        level_precision: float = 2000.0,
        prior_scale: float = 1.0,
        max_goals: int = 10,
    ) -> None:
        super().__init__(xi_candidates=(0.0,), alpha_candidates=(0.0,), max_goals=max_goals, prior_scale=prior_scale)
        self.block_days = block_days
        self.step_precision = step_precision
        self.summer_variance = summer_variance
        self.summer_reversion = summer_reversion
        self.level_precision = level_precision
        # Tiny ridge that pins the otherwise unidentified common shift of attack and defence.
        self.anchor_precision = 0.01

    def fit(self, matches: pd.DataFrame, as_of_date: object) -> "DynamicDixonColesModel":
        data = self._strict_training_data(matches, as_of_date)
        cutoff = pd.Timestamp(as_of_date)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        self.training_rows = len(data)
        self.training_max_date = data["kickoff_utc"].max()
        self.season = season_start_year(cutoff)
        self._derive_promoted_priors(data, self.season)
        if self.split_spells:
            data = self._split_spells(data, self.season)
        self.teams = sorted(set(data["home"]) | set(data["away"]))
        count = len(self.teams)
        origin = data["kickoff_utc"].min().normalize()
        match_block = ((data["kickoff_utc"] - origin).dt.days // self.block_days).to_numpy()
        blocks = int((cutoff - origin).days // self.block_days) + 1
        starts = pd.Series(origin + pd.to_timedelta(np.arange(blocks) * self.block_days, unit="D"))
        block_season = season_labels(starts).to_numpy()
        boundary = np.r_[False, block_season[1:] != block_season[:-1]]

        design = self._design(data, 0.0, 0.0)
        labels = season_labels(data["kickoff_utc"])
        previous = data.loc[labels == self.season - 1]
        established = np.array(sorted({self.teams.index(t) for t in set(previous["home"]) | set(previous["away"])}), dtype=int)
        newcomer_blocks = []
        if len(established) and self.prior_strength > 0:
            current = data.loc[labels == self.season]
            for team in sorted((set(current["home"]) | set(current["away"])) - {self.teams[i] for i in established}):
                rows = current.loc[(current["home"] == team) | (current["away"] == team)]
                first = int(((rows["kickoff_utc"].min() - origin).days) // self.block_days)
                newcomer_blocks.append((self.teams.index(team), first))
        layout = {"blocks": blocks, "count": count, "match_block": match_block, "boundary": boundary,
                  "established": established, "newcomers": newcomer_blocks, "n": len(data)}

        initial = np.zeros(2 * blocks + 2 * blocks * count + 1)
        initial[:blocks] = np.log(max(0.2, (data["home_goals"].mean() + data["away_goals"].mean()) / 2))
        initial[blocks : 2 * blocks] = 0.2
        bounds = [(None, None)] * (len(initial) - 1) + [(-0.2, 0.2)]
        result = minimize(self._objective, initial, args=(design, layout), jac=True, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 3000, "ftol": 1e-11})
        if not np.isfinite(result.fun):
            raise RuntimeError(f"Dynamic Dixon-Coles optimization failed: {result.message}")
        mu, home_adv, attack, defence, self.rho = self._split(result.x, blocks, count)
        last = blocks - 1
        self.mu, self.home_advantage = float(mu[last]), float(home_adv[last])
        self.attack = dict(zip(self.teams, attack[last], strict=True))
        self.defence = dict(zip(self.teams, defence[last], strict=True))
        self.home_delta = {}
        centre_index = established if len(established) else np.arange(count)
        self._centre = (float(attack[last, centre_index].mean()), float(defence[last, centre_index].mean()))
        self._unseen = (self._centre[0] + self.prior_delta[0], self._centre[1] + self.prior_delta[1]) if len(established) else self._centre
        self.promoted_average = self._unseen
        self.promoted_teams = {self.teams[i] for i, _ in newcomer_blocks}
        if self.training_max_date >= cutoff:
            raise AssertionError("Temporal cutoff violation in model fit")
        return self

    @staticmethod
    def _split(params: np.ndarray, blocks: int, count: int):
        mu = params[:blocks]
        home = params[blocks : 2 * blocks]
        size = blocks * count
        attack = params[2 * blocks : 2 * blocks + size].reshape(blocks, count)
        defence = params[2 * blocks + size : 2 * blocks + 2 * size].reshape(blocks, count)
        return mu, home, attack, defence, params[-1]

    def _objective(self, params: np.ndarray, design: dict, layout: dict) -> tuple[float, np.ndarray]:
        blocks, count, b = layout["blocks"], layout["count"], layout["match_block"]
        mu, home_adv, attack, defence, rho = self._split(params, blocks, count)
        h, a = design["home"], design["away"]
        eta_h = mu[b] + home_adv[b] + attack[b, h] - defence[b, a]
        eta_a = mu[b] + attack[b, a] - defence[b, h]
        ll, g_h, g_a, g_rho = dc_terms(eta_h, eta_a, rho, design)
        value = -ll.sum()
        g_mu = -np.bincount(b, g_h + g_a, blocks)
        g_home = -np.bincount(b, g_h, blocks)
        flat = blocks * count
        g_att = -(np.bincount(b * count + h, g_h, flat) + np.bincount(b * count + a, g_a, flat)).reshape(blocks, count)
        g_def = (np.bincount(b * count + a, g_h, flat) + np.bincount(b * count + h, g_a, flat)).reshape(blocks, count)
        g_rho_total = -g_rho.sum()

        # Random walks: team effects (looser and optionally mean-reverting across season boundaries).
        precision = np.where(layout["boundary"][1:], self.step_precision / self.summer_variance, self.step_precision)[:, None]
        phi = np.where(layout["boundary"][1:], self.summer_reversion, 1.0)[:, None]
        for effects, grad in ((attack, g_att), (defence, g_def)):
            step = effects[1:] - phi * effects[:-1]
            value += 0.5 * float((precision * step * step).sum())
            grad[1:] += precision * step
            grad[:-1] -= precision * phi * step
            value += 0.5 * self.anchor_precision * float((effects * effects).sum())
            grad += self.anchor_precision * effects
        for level, grad in ((mu, g_mu), (home_adv, g_home)):
            step = np.diff(level)
            value += 0.5 * self.level_precision * float(step @ step)
            grad[1:] += self.level_precision * step
            grad[:-1] -= self.level_precision * step

        # Promoted-team prior on each newcomer's first block of the current season.
        established = layout["established"]
        if layout["newcomers"]:
            k = self.prior_strength
            for effects, grad, delta in ((attack, g_att, self.prior_delta[0]), (defence, g_def, self.prior_delta[1])):
                for team, block in layout["newcomers"]:
                    residual = effects[block, team] - effects[block, established].mean() - delta
                    value += 0.5 * k * residual * residual
                    grad[block, team] += k * residual
                    grad[block, established] -= k * residual / len(established)

        n = layout["n"]
        gradient = np.concatenate([g_mu, g_home, g_att.ravel(), g_def.ravel(), [g_rho_total]]) / n
        return float(value / n), gradient
