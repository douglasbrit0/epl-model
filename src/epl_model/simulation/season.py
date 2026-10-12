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
    state = {
        team: {"points": 0, "gf": 0, "ga": 0, "played": 0}
        for team in teams
    }
    if current_table is not None:
        for row in current_table.to_dict("records"):
            team = row["team"]
            if team in state:
                state[team].update({key: int(row.get(key, 0)) for key in ("points", "gf", "ga", "played")})
    counts = {team: {"title": 0, "top4": 0, "top6": 0, "relegation": 0, "points": 0} for team in teams}
    for _ in range(simulations):
        table = {team: values.copy() for team, values in state.items()}
        for forecast in forecasts:
            flat = forecast.matrix.reshape(-1)
            index = int(rng.choice(flat.size, p=flat / flat.sum()))
            home_goals, away_goals = np.unravel_index(index, forecast.matrix.shape)
            home, away = forecast.home, forecast.away
            if home not in table or away not in table:
                continue
            h, a = table[home], table[away]
            h["played"] += 1
            a["played"] += 1
            h["gf"] += home_goals
            h["ga"] += away_goals
            a["gf"] += away_goals
            a["ga"] += home_goals
            h["points"] += 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
            a["points"] += 3 if away_goals > home_goals else 1 if away_goals == home_goals else 0
        ranked = sorted(teams, key=lambda team: (table[team]["points"], table[team]["gf"] - table[team]["ga"], table[team]["gf"]), reverse=True)
        for position, team in enumerate(ranked):
            counts[team]["title"] += position == 0
            counts[team]["top4"] += position < 4
            counts[team]["top6"] += position < 6
            counts[team]["relegation"] += position >= len(teams) - relegation_places
            counts[team]["points"] += table[team]["points"]
    return pd.DataFrame(
        [
            {
                "team": team,
                "p_title": values["title"] / simulations,
                "p_top4": values["top4"] / simulations,
                "p_top6": values["top6"] / simulations,
                "p_relegation": values["relegation"] / simulations,
                "exp_points": values["points"] / simulations,
            }
            for team, values in counts.items()
        ]
    ).sort_values("exp_points", ascending=False).reset_index(drop=True)
