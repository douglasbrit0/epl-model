"""E10: player-availability adjustments to expected goals (oracle and previous-match variants).

Production configuration per season (xi, prior_scale from stage 1; xG blend alpha from stage 2) plus
three covariates (own attack missing, opponent defence missing, opponent goalkeeper missing) whose
coefficients are fitted inside the walk-forward likelihood. Covariate penalty precision is
nested-selected; the first evaluated season uses "no covariates".

usage: python exp_availability.py MODE      MODE = oracle | previous
writes e10_<MODE>_grid.parquet (opt = penalty, "none" = no covariates)
"""
import functools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from common import matches
from epl_model.data.fpl_xg import attach_team_xg, team_xg_from_archive
from epl_model.evaluation.backtest import walk_forward
from epl_model.evaluation.tuning import XG_COLUMNS
from epl_model.features.availability import COVARIATES, attach_availability, availability_features, load_player_matches
from epl_model.models import DixonColesModel
from epl_model.seasons import season_start_year

ROOT = Path(__file__).resolve().parents[1]
TUNING = json.load(open(ROOT / "reports" / "tuning.json"))
CHOSEN = {int(k): v for k, v in TUNING["chosen_by_season"].items()}
ALPHA = {int(k): v for k, v in TUNING["alpha_by_season"].items()}
MODE = sys.argv[1]
ARCHIVE = sorted((ROOT / "data" / "raw" / "fpl_archive").glob("*_merged_gw.csv"))
M = attach_team_xg(matches(), team_xg_from_archive(ARCHIVE))
M = attach_availability(M, availability_features(load_player_matches(ARCHIVE), MODE))


class AvailabilityModel(DixonColesModel):
    def __init__(self, ridge):
        covariates = COVARIATES if ridge is not None else ()
        super().__init__(xg_columns=XG_COLUMNS, covariates=covariates, covariate_ridge=ridge or 0.0)

    def fit(self, data, as_of_date):
        season = season_start_year(as_of_date)
        config = CHOSEN.get(season, {"xi": 0.002, "prior_scale": 1.0})
        self.xi_candidates, self.prior_scale = (config["xi"],), config["prior_scale"]
        self.alpha_candidates = (ALPHA.get(season, 0.0),)
        return super().fit(data, as_of_date)


def run(ridge):
    predictions = walk_forward(M, model_factory=functools.partial(AvailabilityModel, ridge))
    predictions["opt"] = "none" if ridge is None else str(ridge)
    return predictions


if __name__ == "__main__":
    start = time.time()
    with ProcessPoolExecutor(4) as executor:
        grid = pd.concat(executor.map(run, [None, 1.0, 10.0, 100.0]))
    grid.to_parquet(f"e10_{MODE}_grid.parquet")
    print(f"{MODE}: 4 configs in {time.time() - start:.0f}s")
