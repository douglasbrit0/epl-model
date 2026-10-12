from pathlib import Path
import functools, itertools, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from common import matches
from epl_model.evaluation.backtest import walk_forward
from epl_model.models.dynamic import DynamicDixonColesModel
from epl_model.seasons import season_start_year
CHOSEN = {int(k): v for k, v in json.load(open(Path(__file__).resolve().parents[1] / "reports" / "tuning.json"))["chosen_by_season"].items()}
class Dyn(DynamicDixonColesModel):
    def fit(self, matches, as_of_date):
        self.prior_scale = CHOSEN.get(season_start_year(as_of_date), {"prior_scale": 1.0})["prior_scale"]
        return super().fit(matches, as_of_date)
M = matches()
def run(cfg):
    p = walk_forward(M, model_factory=functools.partial(Dyn, **cfg))
    return p.assign(opt=json.dumps(cfg, sort_keys=True))
if __name__ == "__main__":
    grid = [dict(step_precision=q, summer_variance=v, summer_reversion=r) for q, v, r in itertools.product([100.0, 300.0, 1000.0], [1.0, 4.0], [1.0, 0.85])]
    t = time.time()
    with ProcessPoolExecutor(8) as ex: out = pd.concat(ex.map(run, grid))
    out.to_parquet("e7_grid.parquet"); print(f"{len(grid)} configs in {time.time()-t:.0f}s")
