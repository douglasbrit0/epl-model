"""Value betting without a model: bet the best available bookmaker price when it beats Pinnacle's fair price.

Pinnacle's de-vigged pre-match probabilities are treated as the truth; a bet is placed when the best
price across bookmakers (football-data.co.uk Max* columns, collected at the same time) exceeds the
fair price by the threshold. Stake 1; settled at that best price.
Caveats: Max* prices may include short-lived or erroneous quotes, and soft bookmakers limit
accounts that do this; results are an upper bound on what is achievable.

usage: python line_shopping.py RAW_DIR DIV
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from epl_model.evaluation.metrics import closing_odds_probabilities
from epl_model.seasons import season_labels

raw_dir, div = Path(sys.argv[1]), sys.argv[2]
rows = []
for path in sorted(raw_dir.glob(f"{div}_*.csv")):
    r = pd.read_csv(path, encoding="latin-1", low_memory=False)
    need = ["PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA", "PSCH", "PSCD", "PSCA", "FTR"]
    if not set(need).issubset(r.columns):
        continue
    r["date"] = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce", utc=True)
    rows.append(r[["date", *need]].dropna())
d = pd.concat(rows, ignore_index=True)
d = d[(d[["PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA"]] > 1).all(axis=1)].reset_index(drop=True)
fair = np.vstack([closing_odds_probabilities(tuple(x), "power") for x in d[["PSH", "PSD", "PSA"]].to_numpy(float)])
best = d[["MaxH", "MaxD", "MaxA"]].to_numpy(float)
close = d[["PSCH", "PSCD", "PSCA"]].to_numpy(float)
won = np.eye(3)[d["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy(int)].astype(bool)
week = (season_labels(d["date"]).astype(str) + "-" + d["date"].dt.isocalendar().week.astype(str)).to_numpy()
print(f"{div}: {len(d)} matches with Pinnacle pre/close and best-price columns ({d.date.min().date()} to {d.date.max().date()})")
rng = np.random.default_rng(9)
for threshold in (0.0, 0.02, 0.04, 0.06):
    pick = fair * best - 1 > threshold
    profit = np.where(pick, np.where(won, best - 1, -1), 0).sum(1)
    bets = int(pick.sum())
    clv = np.log(best[pick] / close[pick]).mean() if bets else np.nan
    weekly = pd.Series(profit).groupby(week).sum().to_numpy()
    sims = weekly[rng.integers(0, len(weekly), (4000, len(weekly)))].sum(1)
    lo, hi = np.quantile(sims, [0.025, 0.975])
    print(f"  edge>{threshold:.0%}: {bets:5d} bets, P&L {profit.sum():+7.1f} [{lo:+.1f}, {hi:+.1f}], ROI {profit.sum() / max(bets, 1):+.1%}, mean CLV vs Pinnacle close {clv:+.2%}")
