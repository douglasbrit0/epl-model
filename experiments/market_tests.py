"""Market-efficiency tests: does the model add information to the betting market, and can a blend win?

T1  Walk-forward blend of market and model log-probabilities (conditional logit, fitted only on
    seasons before the one predicted). A model weight near zero means no independent information.
T2  Flat-stake betting at the *pre-match* Pinnacle price (collected days before kickoff) when the
    blend's expected value clears a threshold, plus closing-line value (CLV) of those bets.
T3  The same for Over/Under 2.5 goals.

usage: python market_tests.py PREDICTIONS.csv RAW_DIR [LABEL]
PREDICTIONS needs fixture_id, season, matchweek, outcome, p_home/p_draw/p_away, p_over_2_5,
home_goals, away_goals (as written by run_backtest.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from epl_model.evaluation.metrics import closing_odds_probabilities

PRED, RAW = Path(sys.argv[1]), Path(sys.argv[2])
LABEL = sys.argv[3] if len(sys.argv) > 3 else "E0"
DIV = sys.argv[4] if len(sys.argv) > 4 else "E0"


def first(frame, *columns):
    """First available odds column, falling back in order; non-prices become NaN."""
    out = pd.Series(np.nan, index=frame.index)
    for column in columns:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            out = out.where(out.notna(), values.where(values > 1))
    return out


def load_odds():
    frames = []
    for path in sorted(RAW.glob(f"{DIV}_*.csv")):
        raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
        date = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", utc=True)
        odds = pd.DataFrame({"fixture_id": "fd:" + date.dt.date.astype(str) + ":" + raw["HomeTeam"].astype(str).str.strip() + ":" + raw["AwayTeam"].astype(str).str.strip()})
        for side, letter in (("home", "H"), ("draw", "D"), ("away", "A")):
            odds[f"pre_{side}"] = first(raw, f"PS{letter}", f"Avg{letter}", f"B365{letter}")
            odds[f"close_{side}"] = first(raw, f"PSC{letter}", f"AvgC{letter}", f"B365C{letter}")
        odds["pre_over"] = first(raw, "P>2.5", "Avg>2.5", "B365>2.5")
        odds["pre_under"] = first(raw, "P<2.5", "Avg<2.5", "B365<2.5")
        odds["close_over"] = first(raw, "PC>2.5", "AvgC>2.5", "B365C>2.5")
        odds["close_under"] = first(raw, "PC<2.5", "AvgC<2.5", "B365C<2.5")
        frames.append(odds)
    return pd.concat(frames).drop_duplicates("fixture_id")


def fair(prices):
    out = np.full(prices.shape, np.nan)
    ok = np.isfinite(prices).all(axis=1)
    out[ok] = np.vstack([closing_odds_probabilities(tuple(row), "power") if len(row) == 3 else _fair2(row) for row in prices[ok]])
    return out


def _fair2(row):
    from scipy.optimize import brentq

    inverse = 1 / np.asarray(row)
    k = brentq(lambda power: np.sum(inverse**power) - 1, 0.01, 20)
    return inverse**k


def block_boot(values, groups, n=4000, seed=3):
    table = pd.DataFrame({"v": values, "g": groups}).dropna().groupby("g").v.agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(table), (n, len(table)))
    sums, counts = table["sum"].to_numpy()[idx].sum(1), table["count"].to_numpy()[idx].sum(1)
    return table["sum"].sum() / table["count"].sum(), *np.quantile(sums / counts, [0.025, 0.975])


def boot_total(values, groups, n=4000, seed=5):
    table = pd.DataFrame({"v": values, "g": groups}).groupby("g").v.sum()
    idx = np.random.default_rng(seed).integers(0, len(table), (n, len(table)))
    return table.sum(), *np.quantile(table.to_numpy()[idx].sum(1), [0.025, 0.975])


def fit_blend(log_market, log_model, outcomes, classes):
    """Conditional logit: score_k = a*log m_k + b*log p_k + c_k (c_0 = 0)."""
    def nll(theta):
        a, b, *c = theta
        scores = a * log_market + b * log_model + np.r_[0.0, c][None, :]
        scores -= scores.max(1, keepdims=True)
        logp = scores - np.log(np.exp(scores).sum(1, keepdims=True))
        return -logp[np.arange(len(outcomes)), outcomes].mean()

    return minimize(nll, np.r_[1.0, 0.0, np.zeros(classes - 1)], method="BFGS").x


def apply_blend(theta, log_market, log_model):
    a, b, *c = theta
    scores = a * log_market + b * log_model + np.r_[0.0, c][None, :]
    scores -= scores.max(1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(1, keepdims=True)


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def rps(p, y):
    one_hot = np.eye(p.shape[1])[y]
    return np.square(np.cumsum(p, 1)[:, :-1] - np.cumsum(one_hot, 1)[:, :-1]).mean(1)


def run_market(d, name, model_cols, pre_cols, close_cols, y, thresholds=(0.0, 0.02, 0.05)):
    model = d[model_cols].to_numpy()
    pre, close = d[pre_cols].to_numpy(float), d[close_cols].to_numpy(float)
    keep = np.isfinite(pre).all(1) & np.isfinite(close).all(1)
    d, model, pre, close, y = d[keep].reset_index(drop=True), model[keep], pre[keep], close[keep], y[keep]
    m_pre, m_close = fair(pre), fair(close)
    groups = (d["season"].astype(str) + "-" + d["matchweek"].astype(str)).to_numpy()
    seasons = sorted(d["season"].unique())
    blend_pre, blend_close = np.full(model.shape, np.nan), np.full(model.shape, np.nan)
    weights = []
    for season in seasons[1:]:
        train, test = (d["season"] < season).to_numpy(), (d["season"] == season).to_numpy()
        for market, out in ((m_pre, blend_pre), (m_close, blend_close)):
            theta = fit_blend(np.log(market[train]), np.log(model[train]), y[train], model.shape[1])
            out[test] = apply_blend(theta, np.log(market[test]), np.log(model[test]))
            if market is m_close:
                weights.append((season, theta[0], theta[1]))
    ev = ~np.isnan(blend_close[:, 0])
    print(f"\n=== {name}: {ev.sum()} matches evaluated (seasons {seasons[1]}-{seasons[-1]}; first season only trains the blend)")
    rows = []
    for label, p in (("model", model), ("pre-match market", m_pre), ("closing market", m_close), ("blend(model, pre)", blend_pre), ("blend(model, close)", blend_close)):
        rows.append((label, logloss(p[ev], y[ev]).mean(), rps(p[ev], y[ev]).mean()))
    print(pd.DataFrame(rows, columns=["forecast", "log_loss", "rps"]).round(4).to_string(index=False))
    for base, blend, label in ((m_close, blend_close, "blend(close) vs closing market"), (m_pre, blend_pre, "blend(pre) vs pre-match market")):
        dl = logloss(blend[ev], y[ev]) - logloss(base[ev], y[ev])
        dr = rps(blend[ev], y[ev]) - rps(base[ev], y[ev])
        print(f"{label}: dLL {block_boot(dl, groups[ev])[0]:+.4f} [{block_boot(dl, groups[ev])[1]:+.4f}, {block_boot(dl, groups[ev])[2]:+.4f}]  "
              f"dRPS {block_boot(dr, groups[ev])[0]:+.4f} [{block_boot(dr, groups[ev])[1]:+.4f}, {block_boot(dr, groups[ev])[2]:+.4f}]")
    w = pd.DataFrame(weights, columns=["season", "market_weight", "model_weight"]).round(3)
    print("blend weights (fit on earlier seasons, closing market):", w.to_dict("records"))

    # Betting at pre-match prices with the pre-match blend (only information available before kickoff).
    print("bets at pre-match Pinnacle price (stake 1) using blend(model, pre); CLV = log(pre price / closing price):")
    for source, probs in (("blend", blend_pre), ("model only", model)):
        for threshold in thresholds:
            value = probs[ev] * pre[ev] - 1
            pick = value > threshold
            outcome_hit = np.eye(model.shape[1])[y[ev]].astype(bool)
            profit = np.where(pick, np.where(outcome_hit, pre[ev] - 1, -1), 0).sum(1)
            bets = int(pick.sum())
            clv = np.log(pre[ev][pick] / close[ev][pick]).mean() if bets else np.nan
            total, lo, hi = boot_total(profit, groups[ev])
            print(f"  {source:10s} EV>{threshold:.0%}: {bets:5d} bets, P&L {total:+7.1f} [{lo:+.1f}, {hi:+.1f}], ROI {total / max(bets, 1):+.1%}, mean CLV {clv:+.3%}")
    return w


if __name__ == "__main__":
    pred = pd.read_csv(PRED)
    pred = pred.loc[pred["season"] >= 2018].merge(load_odds(), on="fixture_id", how="left")
    y1x2 = pred["outcome"].to_numpy(int)
    run_market(pred.copy(), f"{LABEL} 1X2", ["p_home", "p_draw", "p_away"], ["pre_home", "pre_draw", "pre_away"], ["close_home", "close_draw", "close_away"], y1x2)
    pred["p_under_2_5"] = 1 - pred["p_over_2_5"]
    y_ou = ((pred["home_goals"] + pred["away_goals"]) < 3).astype(int).to_numpy()  # 0 = over, 1 = under
    run_market(pred.copy(), f"{LABEL} Over/Under 2.5", ["p_over_2_5", "p_under_2_5"], ["pre_over", "pre_under"], ["close_over", "close_under"], y_ou)
