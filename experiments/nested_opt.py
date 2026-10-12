import sys, numpy as np, pandas as pd
from common import scores
g = pd.read_parquet(sys.argv[1]); default = sys.argv[2]; out_csv = sys.argv[3]
g = g[g.season >= 2018].copy(); g["ll"], _ = scores(g)
by = g.groupby(["opt", "season"]).ll.agg(["sum", "count"]).reset_index()
parts, chosen = [], {}
for s in sorted(g.season.unique()):
    past = by[by.season < s]
    if past.empty: opt = default
    else: a = past.groupby("opt")[["sum", "count"]].sum(); opt = (a["sum"] / a["count"]).idxmin()
    chosen[int(s)] = opt; parts.append(g[(g.season == s) & (g.opt == opt)])
pd.concat(parts).to_csv(out_csv, index=False)
print("chosen:", chosen); print("hindsight:", g.groupby("opt").ll.mean().round(5).to_dict())
