"""E8: replace the Dixon-Coles low-score correction with bivariate Poisson or a Frank copula.

Marginal goal means come from the fitted (E3-config) model; one dependence parameter is fitted by
weighted MLE on the same training data at every refit.
"""
from pathlib import Path
import functools, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import poisson
from common import matches
from epl_model.evaluation.backtest import walk_forward
from epl_model.models import DixonColesModel
from epl_model.seasons import season_start_year
CHOSEN = {int(k): v for k, v in json.load(open(Path(__file__).resolve().parents[1] / "reports" / "tuning.json"))["chosen_by_season"].items()}
G = 11

def bp_grid(lh, la, l3):
    """Bivariate Poisson P(x, y) on a GxG grid, marginal means lh, la (l3 shared)."""
    l1, l2 = np.maximum(lh - l3, 1e-6), np.maximum(la - l3, 1e-6)
    x = np.arange(G)
    lp1 = x[None, :] * np.log(l1)[:, None] - gammaln(x + 1)[None, :]
    lp2 = x[None, :] * np.log(l2)[:, None] - gammaln(x + 1)[None, :]
    out = np.zeros((len(lh), G, G))
    for k in range(G):
        term = np.full((len(lh), G, G), -np.inf)
        a, b = lp1[:, : G - k], lp2[:, : G - k]
        term[:, k:, k:] = a[:, :, None] + b[:, None, :] + k * np.log(max(l3, 1e-12)) - gammaln(k + 1)
        out += np.exp(term) if l3 > 0 or k == 0 else 0
    return out * np.exp(-(l1 + l2 + l3))[:, None, None]

def frank_grid(lh, la, kappa):
    x = np.arange(-1, G)
    F1 = poisson.cdf(x[None, :], lh[:, None]); F2 = poisson.cdf(x[None, :], la[:, None])
    if abs(kappa) < 1e-6:
        C = F1[:, :, None] * F2[:, None, :]
    else:
        C = -np.log1p(np.expm1(-kappa * F1)[:, :, None] * np.expm1(-kappa * F2)[:, None, :] / np.expm1(-kappa)) / kappa
    return np.maximum(C[:, 1:, 1:] - C[:, :-1, 1:] - C[:, 1:, :-1] + C[:, :-1, :-1], 1e-15)

class DepModel(DixonColesModel):
    def __init__(self, dep):
        super().__init__(alpha_candidates=(0.0,)); self.dep = dep; self.theta = 0.0
    def fit(self, data, as_of_date):
        cfg = CHOSEN.get(season_start_year(as_of_date), {"xi": 0.002, "prior_scale": 1.0})
        self.xi_candidates, self.prior_scale = (cfg["xi"],), cfg["prior_scale"]
        super().fit(data, as_of_date)
        train = self._split_spells(self._strict_training_data(data, as_of_date), self.season)
        train = train[(train.home_goals < G) & (train.away_goals < G)]
        lh = np.array([self._pair_rates(h, a)[0] for h, a in zip(train.home, train.away)])
        la = np.array([self._pair_rates(h, a)[1] for h, a in zip(train.home, train.away)])
        age = (train.kickoff_utc.max() - train.kickoff_utc).dt.days.to_numpy(); w = np.exp(-self.xi * age)
        hg, ag = train.home_goals.to_numpy(int), train.away_goals.to_numpy(int); idx = np.arange(len(hg))
        def nll(t):
            P = bp_grid(lh, la, t) if self.dep == "bp" else frank_grid(lh, la, t)
            return -(w * np.log(np.maximum(P[idx, hg, ag], 1e-15))).sum()
        bounds = (0.0, 0.4) if self.dep == "bp" else (-3.0, 3.0)
        self.theta = minimize_scalar(nll, bounds=bounds, method="bounded").x
        return self
    def _pair_rates(self, home, away):
        ha, hd = self.attack.get(home, self._unseen[0]), self.defence.get(home, self._unseen[1])
        aa, ad = self.attack.get(away, self._unseen[0]), self.defence.get(away, self._unseen[1])
        return (np.exp(np.clip(self.mu + self.home_advantage + ha - ad, -5, 4)), np.exp(np.clip(self.mu + aa - hd, -5, 4)))
    def scoreline_matrix(self, home, away, max_goals=None):
        lh, la = self._pair_rates(home, away)
        P = (bp_grid if self.dep == "bp" else frank_grid)(np.array([lh]), np.array([la]), self.theta)[0]
        return P / P.sum()

M = matches()
def run(dep):
    return walk_forward(M, model_factory=functools.partial(DepModel, dep)).assign(opt=dep)
if __name__ == "__main__":
    t = time.time()
    with ProcessPoolExecutor(2) as ex: out = pd.concat(ex.map(run, ["bp", "frank"]))
    out.to_parquet("e8_grid.parquet"); print(f"done in {time.time()-t:.0f}s")
