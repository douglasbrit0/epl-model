"""Nested walk-forward tuning of fixed model configurations.

Each configuration in the grid is run through the full walk-forward. Season S is then predicted with
the configuration that had the best 1X2 log-loss over all evaluated seasons before S, so tuning only
ever sees earlier data. The first evaluated season uses a literature default. Production uses the
configuration that is best over every evaluated season.

Stage 1 tunes decay (xi) and promoted-prior strength. Stage 2 then tunes the goals/xG blend (alpha)
with each season's stage-1 configuration fixed.
"""

from __future__ import annotations

import functools
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from epl_model.evaluation.backtest import walk_forward
from epl_model.models import DixonColesModel

XI_GRID = (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0065)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
XG_COLUMNS = ("fpl_xg_home", "fpl_xg_away")
PRIOR_SCALE_GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
# Dixon & Coles (1997): 0.0065 per half-week, about 0.002 per day.
DEFAULT_CONFIG = {"xi": 0.002, "prior_scale": 1.0}
DEFAULT_ALPHA = 0.0
FIRST_EVALUATED_SEASON = 2018


class SeasonConfigModel(DixonColesModel):
    """Uses the stage-1 (xi, prior_scale) chosen for the season being predicted, and a fixed alpha."""

    def __init__(self, chosen: dict[int, dict], alpha: float):
        super().__init__(alpha_candidates=(alpha,), xg_columns=XG_COLUMNS)
        self.chosen = chosen

    def fit(self, matches: pd.DataFrame, as_of_date: object) -> "SeasonConfigModel":
        from epl_model.seasons import season_start_year

        config = self.chosen.get(season_start_year(as_of_date), DEFAULT_CONFIG)
        self.xi_candidates, self.prior_scale = (config["xi"],), config["prior_scale"]
        return super().fit(matches, as_of_date)


def _run_alpha(matches: pd.DataFrame, chosen: dict[int, dict], alpha: float) -> pd.DataFrame:
    predictions = walk_forward(matches, model_factory=functools.partial(SeasonConfigModel, chosen, alpha))
    return predictions.assign(alpha=alpha)


def run_alpha_grid(matches: pd.DataFrame, chosen: dict[int, dict], workers: int | None = None) -> pd.DataFrame:
    """Stage 2: walk-forward predictions for every xG blend weight, on top of the stage-1 choices."""
    with ProcessPoolExecutor(workers) as executor:
        return pd.concat(executor.map(functools.partial(_run_alpha, matches, chosen), ALPHA_GRID), ignore_index=True)


def _run_config(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    factory = functools.partial(DixonColesModel, xi_candidates=(config["xi"],), alpha_candidates=(0.0,), prior_scale=config["prior_scale"])
    predictions = walk_forward(matches, model_factory=factory)
    return predictions.assign(**config)


def run_grid(matches: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Walk-forward predictions for every grid configuration, run in parallel."""
    configs = [{"xi": xi, "prior_scale": scale} for xi, scale in itertools.product(XI_GRID, PRIOR_SCALE_GRID)]
    with ProcessPoolExecutor(workers) as executor:
        return pd.concat(executor.map(functools.partial(_run_config, matches), configs), ignore_index=True)


def _log_loss(frame: pd.DataFrame) -> np.ndarray:
    probabilities = frame[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = frame["outcome"].to_numpy(dtype=int)
    return -np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-12, 1))


def nested_select(grid: pd.DataFrame, default: dict | None = None) -> tuple[pd.DataFrame, dict[int, dict], dict]:
    """Assemble nested predictions; return them, the per-season choice and the production config.

    The tuned keys are those of ``default`` (stage 1: xi and prior_scale).
    """
    default = default or DEFAULT_CONFIG
    keys = list(default)
    grid = grid.loc[grid["season"] >= FIRST_EVALUATED_SEASON].assign(_ll=lambda frame: _log_loss(frame))
    totals = grid.groupby(keys + ["season"])["_ll"].agg(["sum", "count"]).reset_index()

    def best(seasons: pd.DataFrame) -> dict:
        summed = seasons.groupby(keys)[["sum", "count"]].sum()
        winner = (summed["sum"] / summed["count"]).idxmin()
        winner = winner if isinstance(winner, tuple) else (winner,)  # one tuned key gives a scalar
        return dict(zip(keys, (float(value) for value in winner)))

    chosen, parts = {}, []
    for season in sorted(grid["season"].unique()):
        earlier = totals.loc[totals["season"] < season]
        config = best(earlier) if not earlier.empty else dict(default)
        chosen[int(season)] = config
        mask = grid["season"] == season
        for key, value in config.items():
            mask &= np.isclose(grid[key], value)
        parts.append(grid.loc[mask])
    return pd.concat(parts).drop(columns="_ll"), chosen, best(totals)


def save_tuning(path: Path, chosen: dict[int, dict], production: dict, alpha_by_season: dict[int, dict] | None = None) -> None:
    """Persist the production configuration and the per-season choices."""
    payload = {"production": production, "chosen_by_season": {str(season): config for season, config in chosen.items()}}
    if alpha_by_season is not None:
        payload["alpha_by_season"] = {str(season): config["alpha"] for season, config in alpha_by_season.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_production_config(path: Path) -> dict:
    """Production configuration from the last backtest, or the default if none is saved."""
    if not path.exists():
        return {**DEFAULT_CONFIG, "alpha": DEFAULT_ALPHA}
    production = json.loads(path.read_text(encoding="utf-8"))["production"]
    return {"alpha": DEFAULT_ALPHA, **production}
