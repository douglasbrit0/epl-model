"""Run one experiment: E3 per-season config + an extra model option on a grid, nested-selected.

usage: python exp.py NAME OPTION DEFAULT v1,v2,...   (DEFAULT = the 'off' value, used for the first season)
"""
from pathlib import Path
import functools, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from common import matches
from epl_model.evaluation.backtest import walk_forward
from epl_model.models import DixonColesModel
from epl_model.seasons import season_start_year

CHOSEN = {int(k): v for k, v in json.load(open(Path(__file__).resolve().parents[1] / "reports" / "tuning.json"))["chosen_by_season"].items()}

class SeasonConfigModel(DixonColesModel):
    """Uses the nested-tuned (xi, prior_scale) of the season being predicted."""
    def __init__(self, **extra):
        super().__init__(alpha_candidates=(0.0,), **extra)
    def fit(self, matches, as_of_date):
        cfg = CHOSEN.get(season_start_year(as_of_date), {"xi": 0.002, "prior_scale": 1.0})
        self.xi_candidates, self.prior_scale = (cfg["xi"],), cfg["prior_scale"]
        return super().fit(matches, as_of_date)

M = matches()
def run(args):
    option, value = args
    p = walk_forward(M, model_factory=functools.partial(SeasonConfigModel, **{option: value}))
    return p.assign(opt=str(value))

if __name__ == "__main__":
    name, option, default, values = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    cast = lambda v: None if v == "None" else float(v)
    t = time.time()
    with ProcessPoolExecutor(8) as ex:
        out = pd.concat(ex.map(run, [(option, cast(v)) for v in values]))
    out.to_parquet(f"{name}_grid.parquet"); print(f"{len(values)} values in {time.time()-t:.0f}s")
