"""Paired comparison of two walk-forward prediction files on common fixtures (block bootstrap by matchweek)."""
import sys, numpy as np, pandas as pd
A, B, label = sys.argv[1], sys.argv[2], sys.argv[3]
a = pd.read_csv(A); b = pd.read_csv(B)
x = a.merge(b, on="fixture_id", suffixes=("_a", "_b"))
x = x[x.season_a >= 2018]
y = x.outcome_a.to_numpy(int); oh = np.eye(3)[y]
def ll(s): p = x[[f"p_home_{s}", f"p_draw_{s}", f"p_away_{s}"]].to_numpy(); return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
def rps(s): p = x[[f"p_home_{s}", f"p_draw_{s}", f"p_away_{s}"]].to_numpy(); return np.square(np.cumsum(p, 1)[:, :2] - np.cumsum(oh, 1)[:, :2]).mean(1)
x["dll"] = ll("b") - ll("a"); x["drps"] = rps("b") - rps("a"); x["ll_a"], x["ll_b"], x["rps_a"], x["rps_b"] = ll("a"), ll("b"), rps("a"), rps("b")
x["wk"] = x.season_a.astype(str) + "-" + x.matchweek_a.astype(str)
def boot(v, g, n=4000):
    t = pd.DataFrame({"v": v, "g": g}).groupby("g").v.agg(["sum", "count"]); idx = np.random.default_rng(11).integers(0, len(t), (n, len(t)))
    s = t["sum"].to_numpy()[idx].sum(1) / t["count"].to_numpy()[idx].sum(1)
    return t["sum"].sum() / t["count"].sum(), *np.quantile(s, [.025, .975]), (s >= 0).mean()
rows = []
for s, g in list(x.groupby("season_a")) + [("ALL", x), ("2023-25", x[x.season_a.between(2023, 2025)])]:
    m1, l1, h1, p1 = boot(g.dll, g.wk); m2, l2, h2, p2 = boot(g.drps, g.wk)
    rows.append(dict(season=s, n=len(g), ll_a=g.ll_a.mean(), ll_b=g.ll_b.mean(), d_ll=m1, d_ll_lo=l1, d_ll_hi=h1, p_not_better_ll=p1,
                     rps_a=g.rps_a.mean(), rps_b=g.rps_b.mean(), d_rps=m2, d_rps_lo=l2, d_rps_hi=h2, p_not_better_rps=p2))
r = pd.DataFrame(rows); r.to_csv(f"compare_{label}.csv", index=False)
pd.set_option("display.width", 250); print(r.round(4).to_string(index=False))
