"""Strict matchweek walk-forward evaluation and diagnostic flat-stake edges."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epl_model.evaluation.metrics import closing_odds_probabilities, multiclass_scores, outcome_probabilities
from epl_model.models import DixonColesModel


def walk_forward(matches: pd.DataFrame, model_factory=DixonColesModel) -> pd.DataFrame:
    """Predict matchweeks using only matches before that week's first kickoff.

    Historical E0 files do not consistently include matchweek numbers. When absent,
    the walk-forward blocks are Monday-start calendar weeks within each EPL season.
    """
    data = matches.copy()
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True)
    data = data.sort_values("kickoff_utc")
    data["season"] = data["kickoff_utc"].map(lambda value: value.year if value.month >= 7 else value.year - 1)
    if "matchweek" in data and data["matchweek"].notna().all():
        data["walkweek"] = data["matchweek"].astype(int)
    else:
        season_start = pd.to_datetime(data["season"].astype(str) + "-07-01", utc=True)
        data["walkweek"] = ((data["kickoff_utc"] - season_start).dt.days // 7).astype(int) + 1
    records = []
    for (season, week), group in data.groupby(["season", "walkweek"], sort=True):
        cutoff = group["kickoff_utc"].min().normalize()
        training = data.loc[data["kickoff_utc"] < cutoff]
        if len(training) < 30:
            continue
        model = model_factory().fit(training, cutoff)
        if model.training_max_date >= cutoff:
            raise AssertionError("Walk-forward leakage: training reached the prediction date")
        forecasts = model.predict(group)
        for fixture, forecast in zip(group.to_dict("records"), forecasts, strict=True):
            probs = outcome_probabilities(forecast.matrix)
            outcome = 0 if fixture["home_goals"] > fixture["away_goals"] else 1 if fixture["home_goals"] == fixture["away_goals"] else 2
            record = {
                "fixture_id": fixture["fixture_id"],
                "kickoff_utc": fixture["kickoff_utc"],
                "outcome": outcome,
                "p_home": probs[0],
                "p_draw": probs[1],
                "p_away": probs[2],
                "season": int(season),
                "matchweek": int(week),
                "fit_data_through": model.training_max_date.isoformat(),
            }
            for key, value in zip(("odds_home", "odds_draw", "odds_away"), (fixture.get("odds_home"), fixture.get("odds_draw"), fixture.get("odds_away")), strict=True):
                record[key] = value
            records.append(record)
    return pd.DataFrame(records)


def flat_stake_profit(predictions: pd.DataFrame, threshold: float = 0.02) -> np.ndarray:
    """Return realized unit P&L per match for any side at least threshold above fair price."""
    profits = np.zeros(len(predictions), dtype=float)
    for position, row in enumerate(predictions.itertuples(index=False)):
        prices = np.array([row.odds_home, row.odds_draw, row.odds_away], dtype=float)
        if not np.isfinite(prices).all() or (prices <= 1).any():
            continue
        fair = closing_odds_probabilities(tuple(prices), "power")
        model_probs = np.array([row.p_home, row.p_draw, row.p_away])
        for outcome in np.flatnonzero(model_probs - fair >= threshold):
            profits[position] += prices[outcome] - 1 if row.outcome == outcome else -1
    return profits


def summarize_walk_forward(predictions: pd.DataFrame) -> dict[str, float]:
    """Summarize model and de-vigged closing-market scoring metrics."""
    p = predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    y = predictions["outcome"].to_numpy(dtype=int)
    output = {f"model_{key}": value for key, value in multiclass_scores(p, y).items()}
    valid = predictions[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1)
    if valid.any():
        markets = np.vstack([
            closing_odds_probabilities(tuple(row), method="power")
            for row in predictions.loc[valid, ["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
        ])
        output.update({f"market_power_{key}": value for key, value in multiclass_scores(markets, y[valid.to_numpy()]).items()})
        shin = np.vstack([
            closing_odds_probabilities(tuple(row), method="shin")
            for row in predictions.loc[valid, ["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
        ])
        output.update({f"market_shin_{key}": value for key, value in multiclass_scores(shin, y[valid.to_numpy()]).items()})
    return output


def flat_stake_edges(predictions: pd.DataFrame, thresholds=range(2, 11), bootstrap_samples: int = 2000, seed: int = 1729) -> pd.DataFrame:
    """Compute flat-unit P&L on positive model edges and bootstrap 95% intervals."""
    rng = np.random.default_rng(seed)
    records = []
    for row in predictions.itertuples(index=False):
        prices = np.array([row.odds_home, row.odds_draw, row.odds_away], dtype=float)
        if not np.isfinite(prices).all() or (prices <= 1).any():
            continue
        fair = closing_odds_probabilities(tuple(prices), "power")
        model_probs = np.array([row.p_home, row.p_draw, row.p_away])
        for outcome, side in enumerate(("home", "draw", "away")):
            edge = model_probs[outcome] - fair[outcome]
            if edge <= 0:
                continue
            won = int(row.outcome == outcome)
            records.append({"outcome": outcome, "side": side, "edge": edge, "odds": prices[outcome], "profit": prices[outcome] - 1 if won else -1})
    frame = pd.DataFrame(records)
    output = []
    for threshold in thresholds:
        selected = frame.loc[frame["edge"] >= threshold / 100, "profit"].to_numpy() if not frame.empty else np.array([])
        if len(selected):
            draws = rng.choice(selected, size=(bootstrap_samples, len(selected)), replace=True).sum(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            pnl = float(selected.sum())
        else:
            low = high = pnl = 0.0
        output.append({"edge_threshold_pct": threshold, "bets": int(len(selected)), "flat_stake_pnl": pnl, "ci_95_low": float(low), "ci_95_high": float(high)})
    return pd.DataFrame(output)
