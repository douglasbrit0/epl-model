"""Monte Carlo remaining-season outcomes with Premier League tiebreakers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_season(
    forecasts,
    teams: list[str],
    current_table: pd.DataFrame | None = None,
    simulations: int = 10_000,
    seed: int = 1729,
    relegation_places: int = 3,
) -> pd.DataFrame:
    """Simulate remaining fixtures and rank by points, goal difference, goals scored."""
    rng = np.random.default_rng(seed)
    index = {team: i for i, team in enumerate(teams)}
    count = len(teams)
    points = np.zeros((simulations, count), dtype=np.int64)
    goals_for = np.zeros((simulations, count), dtype=np.int64)
    goals_against = np.zeros((simulations, count), dtype=np.int64)
    if current_table is not None:
        for row in current_table.to_dict("records"):
            if row["team"] in index:
                position = index[row["team"]]
                points[:, position] = int(row.get("points", 0))
                goals_for[:, position] = int(row.get("gf", 0))
                goals_against[:, position] = int(row.get("ga", 0))
    for forecast in forecasts:
        if forecast.home not in index or forecast.away not in index:
            continue
        home, away = index[forecast.home], index[forecast.away]
        flat = forecast.matrix.reshape(-1)
        draws = rng.choice(flat.size, size=simulations, p=flat / flat.sum())
        home_goals, away_goals = np.unravel_index(draws, forecast.matrix.shape)
        goals_for[:, home] += home_goals
        goals_against[:, home] += away_goals
        goals_for[:, away] += away_goals
        goals_against[:, away] += home_goals
        points[:, home] += np.where(home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0))
        points[:, away] += np.where(away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0))
    # Points, then goal difference, then goals scored; ties keep the input team order.
    difference = goals_for - goals_against
    key = points * 10**8 + (difference + 10**4) * 10**3 + np.minimum(goals_for, 999)
    order = np.argsort(-key, axis=1, kind="stable")
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(count)[None, :].repeat(simulations, axis=0), axis=1)
    return pd.DataFrame(
        {
            "team": teams,
            "p_title": (position == 0).mean(axis=0),
            "p_top4": (position < 4).mean(axis=0),
            "p_top6": (position < 6).mean(axis=0),
            "p_relegation": (position >= count - relegation_places).mean(axis=0),
            "exp_points": points.mean(axis=0),
        }
    ).sort_values("exp_points", ascending=False).reset_index(drop=True)
