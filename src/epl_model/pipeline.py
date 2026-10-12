"""Validated model selection and daily contract payload construction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from epl_model.evaluation.metrics import closing_odds_probabilities, outcome_probabilities, score_probabilities
from epl_model.models import DixonColesModel
from epl_model.seasons import season_start_year
from epl_model.simulation.season import simulate_season


def choose_validated_model(matches: pd.DataFrame, seed: int = 1729) -> DixonColesModel:
    """Select goals-only vs xG-blended DC by a chronological held-out log-loss."""
    data = matches.sort_values("kickoff_utc").reset_index(drop=True)
    split = max(1, int(len(data) * 0.85))
    validation = data.iloc[split:]
    training = data.iloc[:split]
    if len(training) < 100 or validation.empty:
        raise ValueError("At least 100 training matches and a chronological validation tail are required")
    cutoff = pd.to_datetime(validation["kickoff_utc"], utc=True).min()
    specs = [("dc-goals", (0.0,))]
    if {"npxg_home", "npxg_away"}.issubset(data.columns) and data[["npxg_home", "npxg_away"]].notna().all().all():
        specs.append(("dc-xg-blended", (0.0, 0.25, 0.5, 0.75, 1.0)))
    scores = []
    for version, alphas in specs:
        model = DixonColesModel(xi_candidates=(0.0, 0.001, 0.002, 0.003, 0.005), alpha_candidates=alphas)
        model.fit(training, cutoff)
        forecasts = model.predict(validation)
        losses = []
        for row, forecast in zip(validation.to_dict("records"), forecasts, strict=True):
            probabilities = outcome_probabilities(forecast.matrix)
            outcome = 0 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 2
            losses.append(-np.log(max(probabilities[outcome], 1e-12)))
        scores.append((float(np.mean(losses)), version, alphas))
    _, version, alphas = min(scores, key=lambda value: value[0])
    prediction_date = datetime.now(timezone.utc).date()
    final = DixonColesModel(alpha_candidates=alphas).fit(data, pd.Timestamp(prediction_date, tz="UTC"))
    final.model_version = f"{version}-xi{final.xi:.4f}-alpha{final.alpha:.2f}"
    return final


def current_table(matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Build current points/GF/GA from completed matches in the current season."""
    data = matches.copy()
    dates = pd.to_datetime(data["kickoff_utc"], utc=True)
    today = pd.Timestamp.now(tz="UTC")
    season_start = pd.Timestamp(year=season_start_year(today), month=7, day=1, tz="UTC")
    season = data.loc[(dates >= season_start) & (dates < today)]
    records = {team: {"points": 0, "gf": 0, "ga": 0, "played": 0} for team in teams}
    for row in season.to_dict("records"):
        if row["home"] not in records or row["away"] not in records:
            continue
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        home, away = records[row["home"]], records[row["away"]]
        home["played"] += 1
        away["played"] += 1
        home["gf"] += hg
        home["ga"] += ag
        away["gf"] += ag
        away["ga"] += hg
        home["points"] += 3 if hg > ag else 1 if hg == ag else 0
        away["points"] += 3 if ag > hg else 1 if hg == ag else 0
    return pd.DataFrame([{"team": team, **values} for team, values in records.items()])


def _fixture_payload(forecast, fixture: dict, model_version: str) -> dict:
    """Translate full model probabilities into the stable DougBoard fixture shape."""
    matrix = forecast.matrix
    markets = score_probabilities(matrix)
    compact = matrix[:7, :7].copy()
    top_indices = np.argsort(matrix, axis=None)[-5:][::-1]
    top = []
    for flat_index in top_indices:
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        top.append({"home_goals": int(home_goals), "away_goals": int(away_goals), "probability": float(matrix[home_goals, away_goals])})
    payload = {
        "fixture_id": forecast.fixture_id,
        "kickoff_utc": pd.Timestamp(fixture["kickoff_utc"]).isoformat().replace("+00:00", "Z"),
        "home": forecast.home,
        "away": forecast.away,
        **markets,
        "xg_home": forecast.xg_home,
        "xg_away": forecast.xg_away,
        "top_scorelines": top,
        "scoreline_matrix": compact.tolist(),
    }
    odds = [fixture.get("odds_home"), fixture.get("odds_draw"), fixture.get("odds_away")]
    if all(value is not None and pd.notna(value) and float(value) > 1 for value in odds):
        fair = closing_odds_probabilities(tuple(float(value) for value in odds), "power")
        payload.update(
            {
                "market_p_home": float(fair[0]),
                "market_p_draw": float(fair[1]),
                "market_p_away": float(fair[2]),
                "edge_home": markets["p_home"] - float(fair[0]),
                "edge_draw": markets["p_draw"] - float(fair[1]),
                "edge_away": markets["p_away"] - float(fair[2]),
                "odds_source": fixture.get("odds_source", "unknown"),
            }
        )
    return payload


def make_payloads(model: DixonColesModel, matches: pd.DataFrame, fixtures: pd.DataFrame, season_fixtures: pd.DataFrame, generated_at: datetime | None = None, previous_payloads: dict | None = None) -> tuple[dict, list[dict]]:
    """Build contract payloads and immutable log entries before publication."""
    generated = generated_at or datetime.now(timezone.utc)
    generated_iso = generated.isoformat().replace("+00:00", "Z")
    model_version = getattr(model, "model_version", f"dc-xi{model.xi:.4f}-alpha{model.alpha:.2f}")
    forecasts = model.predict(fixtures)
    fixture_records = fixtures.to_dict("records")
    fixture_payloads = [_fixture_payload(forecast, row, model_version) for forecast, row in zip(forecasts, fixture_records, strict=True)]
    log_records = []
    for payload in fixture_payloads:
        import hashlib

        prediction_id = hashlib.sha256(f"{payload['fixture_id']}|{model_version}|{generated_iso}".encode()).hexdigest()
        log_records.append({
            "prediction_id": prediction_id,
            "fixture_id": payload["fixture_id"],
            "model_version": model_version,
            "generated_at": generated_iso,
            "kickoff_utc": payload["kickoff_utc"],
            "home": payload["home"],
            "away": payload["away"],
            "p_home": payload["p_home"], "p_draw": payload["p_draw"], "p_away": payload["p_away"],
            "xg_home": payload["xg_home"], "xg_away": payload["xg_away"],
            **({"market_p_home": payload["market_p_home"], "market_p_draw": payload["market_p_draw"], "market_p_away": payload["market_p_away"]} if "market_p_home" in payload else {}),
        })
    season_forecasts = model.predict(season_fixtures)
    match_dates = pd.to_datetime(matches["kickoff_utc"], utc=True)
    today = pd.Timestamp.now(tz="UTC")
    season_start = pd.Timestamp(year=season_start_year(today), month=7, day=1, tz="UTC")
    current_results = matches.loc[(match_dates >= season_start) & (match_dates < today)]
    teams = sorted(
        set(current_results["home"])
        | set(current_results["away"])
        | set(season_fixtures["home"])
        | set(season_fixtures["away"])
    )
    table = current_table(matches, teams)
    season_df = simulate_season(season_forecasts, teams, table, simulations=10_000)
    season_rows = season_df.to_dict("records")
    rating_rows = [
        {"team": team, "attack": float(model.attack.get(team, 0)), "defence": float(model.defence.get(team, 0)), "overall": float(model.attack.get(team, 0) + model.defence.get(team, 0))}
        for team in teams
    ]
    data_through = pd.to_datetime(matches["kickoff_utc"], utc=True).max().date().isoformat()
    history_date = datetime.now(timezone.utc).date().isoformat()
    ratings_history = list((previous_payloads or {}).get("ratings.json", {}).get("history", []))
    ratings_history.extend({"date": history_date, **rating} for rating in rating_rows)
    ratings_history = list({(row["date"], row["team"]): row for row in ratings_history}.values())
    played_by_team = table.set_index("team")["played"].to_dict()
    season_history = list((previous_payloads or {}).get("season.json", {}).get("history", []))
    season_history.extend({"matchweek": int(played_by_team.get(row["team"], 0)), **row} for row in season_rows)
    season_history = list({(row["matchweek"], row["team"]): row for row in season_history}.values())
    performance = _performance_payload(model_version)
    payloads = {
        "latest.json": {"generated_at": generated_iso, "model_version": model_version, "data_through": data_through},
        "fixtures.json": fixture_payloads,
        "ratings.json": {"generated_at": generated_iso, "model_version": model_version, "teams": rating_rows, "history": ratings_history},
        "season.json": {"generated_at": generated_iso, "model_version": model_version, "teams": season_rows, "history": season_history},
        "performance.json": performance,
    }
    return payloads, log_records


def backtest_summary(predictions: pd.DataFrame) -> dict:
    """Condense walk-forward predictions into the backtest part of performance.json."""
    from epl_model.evaluation.metrics import closing_odds_probabilities, multiclass_scores, reliability_bins

    window = predictions.tail(250)
    metrics = multiclass_scores(window[["p_home", "p_draw", "p_away"]].to_numpy(), window["outcome"].to_numpy(dtype=int))
    market_log_loss = market_rps = None
    odds_columns = ["odds_home", "odds_draw", "odds_away"]
    if set(odds_columns).issubset(window.columns):
        valid = window[odds_columns].notna().all(axis=1)
        if valid.any():
            market = np.vstack([closing_odds_probabilities(tuple(values), "power") for values in window.loc[valid, odds_columns].to_numpy(dtype=float)])
            market_scores = multiclass_scores(market, window.loc[valid, "outcome"].to_numpy(dtype=int))
            market_log_loss, market_rps = market_scores["log_loss"], market_scores["rps"]
    return {
        "rolling": {
            "window_end": pd.to_datetime(window["kickoff_utc"], utc=True).max().date().isoformat(),
            "model_log_loss": metrics["log_loss"],
            "model_rps": metrics["rps"],
            "market_log_loss": market_log_loss,
            "market_rps": market_rps,
        },
        "calibration_bins": reliability_bins(predictions[["p_home", "p_draw", "p_away"]].to_numpy(), predictions["outcome"].to_numpy(dtype=int)),
        "flat_stake_pnl": float(predictions["flat_stake_profit"].fillna(0).sum()) if "flat_stake_profit" in predictions else 0.0,
    }


def _performance_payload(model_version: str) -> dict:
    """Load available walk-forward evidence; empty arrays are honest before backtest.

    Prefers the committed ``reports/backtest_summary.json`` (the full CSV is not committed,
    so it is absent on the daily runner) and falls back to ``reports/walk_forward.csv``.
    """
    summary_path = Path("reports/backtest_summary.json")
    report = Path("reports/walk_forward.csv")
    rolling = []
    bins = []
    pnl = 0.0
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    elif report.exists():
        predictions = pd.read_csv(report)
        if not predictions.empty and {"outcome", "p_home", "p_draw", "p_away"}.issubset(predictions.columns):
            summary = backtest_summary(predictions)
    predictions = pd.read_csv(report) if report.exists() else pd.DataFrame()
    if summary is not None:
        rolling.append(summary["rolling"])
        bins = summary["calibration_bins"]
        pnl = summary["flat_stake_pnl"]
    scored_path = Path("logs/scored_predictions.jsonl")
    prediction_log_path = Path("logs/prediction_log.jsonl")
    if scored_path.exists() and prediction_log_path.exists():
        scored = pd.read_json(scored_path, lines=True)
        logged = pd.read_json(prediction_log_path, lines=True)
        live = scored.merge(logged, on=["prediction_id", "fixture_id"], suffixes=("", "_prediction"))
        if not live.empty:
            if "fixture_id" in predictions:
                live = live.loc[~live["fixture_id"].isin(predictions["fixture_id"])]
        if not live.empty:
            window = live.tail(250)
            rolling.append({
                "window_end": pd.to_datetime(window["generated_at"], utc=True).max().date().isoformat(),
                "model_log_loss": float(window["model_log_loss"].mean()),
                "model_rps": float(window["model_rps"].mean()),
                "market_log_loss": float(window["market_log_loss"].mean()) if "market_log_loss" in window and window["market_log_loss"].notna().any() else None,
                "market_rps": float(window["market_rps"].mean()) if "market_rps" in window and window["market_rps"].notna().any() else None,
            })
            if {"actual_outcome", "p_home", "p_draw", "p_away"}.issubset(live.columns):
                from epl_model.evaluation.metrics import reliability_bins

                bins = reliability_bins(live[["p_home", "p_draw", "p_away"]].to_numpy(), live["actual_outcome"].to_numpy(dtype=int))
            if "flat_stake_profit" in live:
                pnl += float(live["flat_stake_profit"].fillna(0).sum())
    return {"generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "model_version": model_version, "rolling": rolling, "calibration_bins": bins, "cumulative_flat_stake_pnl": pnl}
