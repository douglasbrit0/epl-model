"""E9: blend goals with FPL team xG in the training target, on top of the per-season E3 config.

Each alpha is a fixed configuration run through the full walk-forward; nested selection then picks
alpha per season from earlier seasons only. xG exists from 2022/23, so earlier matches train on goals.

usage: python exp_xg.py      (then nested_opt.py e9_grid.parquet 0.0 e9.csv)
"""
import functools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from common import matches
from epl_model.data.fpl_xg import attach_team_xg, team_xg_from_archive
from epl_model.evaluation.backtest import walk_forward
from epl_model.models import DixonColesModel
from epl_model.seasons import season_start_year

ROOT = Path(__file__).resolve().parents[1]
CHOSEN = {int(k): v for k, v in json.load(open(ROOT / "reports" / "tuning.json"))["chosen_by_season"].items()}
M = attach_team_xg(matches(), team_xg_from_archive(sorted((ROOT / "data" / "raw" / "fpl_archive").glob("*_merged_gw.csv"))))


class XgModel(DixonColesModel):
    def __init__(self, alpha):
        super().__init__(alpha_candidates=(alpha,), xg_columns=("fpl_xg_home", "fpl_xg_away"))

    def fit(self, data, as_of_date):
        config = CHOSEN.get(season_start_year(as_of_date), {"xi": 0.002, "prior_scale": 1.0})
        self.xi_candidates, self.prior_scale = (config["xi"],), config["prior_scale"]
        return super().fit(data, as_of_date)


def run(alpha):
    return walk_forward(M, model_factory=functools.partial(XgModel, alpha)).assign(opt=str(alpha))


if __name__ == "__main__":
    start = time.time()
    with ProcessPoolExecutor(8) as executor:
        grid = pd.concat(executor.map(run, [0.0, 0.25, 0.5, 0.75, 1.0]))
    grid.to_parquet("e9_grid.parquet")
    print(f"5 alphas in {time.time() - start:.0f}s")
