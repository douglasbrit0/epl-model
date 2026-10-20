"""Shared helpers for Phase 2 experiments (repo code only; repo name map)."""
from pathlib import Path
import numpy as np, pandas as pd
from epl_model.data.football_data import load_history
from epl_model.data.names import normalize_match_teams

def matches():
    return normalize_match_teams(load_history(sorted((Path(__file__).resolve().parents[1] / "data" / "raw").glob("E0_*.csv"))), "football_data")

def scores(df):
    y = df.outcome.to_numpy(int); p = df[["p_home", "p_draw", "p_away"]].to_numpy(); oh = np.eye(3)[y]
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)); rps = np.square(np.cumsum(p, 1)[:, :2] - np.cumsum(oh, 1)[:, :2]).mean(1)
    return ll, rps
