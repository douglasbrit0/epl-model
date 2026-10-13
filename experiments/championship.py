"""Test 4: the production model on the Championship (football-data.co.uk E1), nested-tuned.

Championship newcomers arrive from two directions: relegated from the Premier League (strong) and
promoted from League One (weak). Each group gets its own empirical-Bayes prior. Whether a club was
relegated is known from the previous season's Premier League table, so it is point-in-time.
Team names are football-data.co.uk spellings (no canonical mapping needed within one source).

usage: python championship.py      (writes championship_walk_forward.csv here)
"""

import functools
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from epl_model.data.football_data import load_history
from epl_model.evaluation.backtest import walk_forward
from epl_model.evaluation.tuning import nested_select
from epl_model.models import DixonColesModel
from epl_model.seasons import season_labels

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
E1 = load_history(sorted(RAW.glob("E1_*.csv")))
E0 = load_history(sorted(RAW.glob("E0_*.csv")))
E0_TEAMS = {int(s): set(g["home"]) | set(g["away"]) for s, g in E0.groupby(season_labels(E0["kickoff_utc"]))}
# Season S -> clubs that played in the Premier League in S-1.
RELEGATED_FROM_ABOVE = {season + 1: teams for season, teams in E0_TEAMS.items()}


def base_name(team: str) -> str:
    return team.replace(" (earlier spell)", "")


class TwoWayPriorModel(DixonColesModel):
    """Separate newcomer priors for clubs coming down from the league above and up from below."""

    def _derive_promoted_priors(self, data, season):
        self.group_priors = {}
        labels = season_labels(data["kickoff_utc"])
        ratings = {"down": [], "up": []}
        for past in sorted(set(labels)):
            past = int(past)
            season_data, previous = data.loc[labels == past], data.loc[labels == past - 1]
            if past >= season or previous.empty or len(season_data) < self.min_season_matches:
                continue
            newcomers = sorted((set(season_data["home"]) | set(season_data["away"])) - (set(previous["home"]) | set(previous["away"])))
            if not newcomers:
                continue
            model = DixonColesModel(xi_candidates=(0.0,), alpha_candidates=(0.0,))
            model._assign_parameters(model._fit_parameters(season_data, xi=0.0, alpha=0.0))
            for team in newcomers:
                group = "down" if team in RELEGATED_FROM_ABOVE.get(past, set()) else "up"
                ratings[group].append((model.attack[team], model.defence[team]))
        for group, values in ratings.items():
            if values:
                values = np.array(values)
                variance = float(np.mean(np.square(values - values.mean(axis=0))))
                self.group_priors[group] = (float(values[:, 0].mean()), float(values[:, 1].mean()), self.prior_scale / max(variance, 0.01))
            else:
                self.group_priors[group] = (0.0, 0.0, 0.0)
        self.prior_delta = self.group_priors["up"][:2]
        self.prior_strength = self.group_priors["up"][2]

    def _promotion_penalty(self, data, season):
        penalty = super()._promotion_penalty(data, season)
        if penalty is None:
            return None
        relegated = RELEGATED_FROM_ABOVE.get(season, set())
        groups = ["down" if base_name(self.teams[i]) in relegated and "spell" not in self.teams[i] else "up" for i in penalty["newcomers"]]
        penalty["delta"] = (np.array([self.group_priors[g][0] for g in groups]), np.array([self.group_priors[g][1] for g in groups]))
        penalty["strength"] = np.array([self.group_priors[g][2] for g in groups])
        return penalty

    def fit(self, matches, as_of_date):
        super().fit(matches, as_of_date)
        down = self.group_priors.get("down", (0.0, 0.0, 0.0))
        self._unseen_by_team = {team: (self._centre[0] + down[0], self._centre[1] + down[1]) for team in RELEGATED_FROM_ABOVE.get(self.season, set())}
        return self


def run(config):
    factory = functools.partial(TwoWayPriorModel, xi_candidates=(config["xi"],), alpha_candidates=(0.0,), prior_scale=config["prior_scale"])
    return walk_forward(E1, model_factory=factory).assign(**config)


if __name__ == "__main__":
    grid = [{"xi": xi, "prior_scale": scale} for xi, scale in itertools.product((0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004), (0.25, 0.5, 1.0, 2.0))]
    start = time.time()
    with ProcessPoolExecutor(8) as executor:
        predictions = pd.concat(executor.map(run, grid), ignore_index=True)
    nested, chosen, production = nested_select(predictions)
    nested.to_csv("championship_walk_forward.csv", index=False)
    print(f"{len(grid)} configs in {time.time() - start:.0f}s; chosen {chosen}; production {production}")
