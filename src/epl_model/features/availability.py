"""Point-in-time player-availability features from FPL per-match player data.

For each team and match, the previous ``window`` team matches (crossing seasons) define the regulars:
players with at least ``regular_share`` of the available minutes. A regular's attacking value is their
share of the team's FPL threat + creativity over that window (FPL's xG-like chance measures, available
for every season from 2020/21).

Three features per team-match:
- ``att_missing``: summed attacking share of regulars (excluding goalkeepers) who are missing.
- ``def_missing``: share of regular defenders' minutes that are missing.
- ``gk_missing``: 1 when the regular goalkeeper is missing.

"Missing" depends on the mode:
- ``oracle``: the player did not play in this match (uses the actual lineup: an upper bound, not
  achievable before kickoff).
- ``previous``: the player did not play in the team's previous match (known before kickoff).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from epl_model.data.names import normalize_team_names

FEATURES = ("att_missing", "def_missing", "gk_missing")


def load_player_matches(paths: list[Path]) -> pd.DataFrame:
    """Long table of player-match rows from the FPL archive (seasons with team and position)."""
    frames = []
    for path in sorted(paths):
        raw = pd.read_csv(path, low_memory=False)
        if not {"team", "position", "minutes", "threat", "creativity", "kickoff_time", "name"}.issubset(raw.columns):
            continue
        frames.append(pd.DataFrame({
            "team": normalize_team_names(raw["team"], "fpl"),
            "date": pd.to_datetime(raw["kickoff_time"], utc=True),
            "name": raw["name"].astype(str),
            "position": raw["position"].replace({"GKP": "GK"}),
            "minutes": pd.to_numeric(raw["minutes"], errors="coerce").fillna(0.0),
            "attack": pd.to_numeric(raw["threat"], errors="coerce").fillna(0.0) + pd.to_numeric(raw["creativity"], errors="coerce").fillna(0.0),
        }))
    if not frames:
        return pd.DataFrame(columns=["team", "date", "name", "position", "minutes", "attack"])
    players = pd.concat(frames, ignore_index=True)
    # A player may appear twice for one match in rare archive duplicates; keep one row.
    return players.drop_duplicates(["team", "date", "name"])


def availability_features(players: pd.DataFrame, mode: str = "oracle", window: int = 8, regular_share: float = 0.5) -> pd.DataFrame:
    """Per (team, match date) availability features; matches without enough history get 0."""
    if mode not in {"oracle", "previous"}:
        raise ValueError("mode must be 'oracle' or 'previous'")
    rows = []
    for team, group in players.groupby("team"):
        minutes = group.pivot_table(index="date", columns="name", values="minutes", aggfunc="sum").sort_index().fillna(0.0)
        attack = group.pivot_table(index="date", columns="name", values="attack", aggfunc="sum").reindex(minutes.index).fillna(0.0)
        position = group.sort_values("date").groupby("name")["position"].last().reindex(minutes.columns)
        values_min, values_att = minutes.to_numpy(), attack.to_numpy()
        is_gk = (position == "GK").to_numpy()
        is_def = (position == "DEF").to_numpy()
        for i, date in enumerate(minutes.index):
            features = dict.fromkeys(FEATURES, 0.0)
            if i >= window:
                past_min, past_att = values_min[i - window : i], values_att[i - window : i]
                share = past_min.sum(axis=0) / (90.0 * window)
                regular = share >= regular_share
                current = values_min[i] if mode == "oracle" else values_min[i - 1]
                missing = regular & (current == 0)
                attack_total = past_att.sum()
                if attack_total > 0:
                    attack_share = past_att.sum(axis=0) / attack_total
                    features["att_missing"] = float(attack_share[missing & ~is_gk].sum())
                defenders = regular & is_def
                if defenders.any():
                    features["def_missing"] = float(share[missing & is_def].sum() / share[defenders].sum())
                keepers = regular & is_gk
                features["gk_missing"] = float(keepers.any() and missing[keepers].all())
            rows.append({"team": team, "date": date, **features})
    return pd.DataFrame(rows)


def attach_availability(matches: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Add model covariates oriented to each side's scoring rate.

    ``home_att_missing`` is the home team's own missing attack; ``home_opp_def_missing`` and
    ``home_opp_gk_missing`` are the away team's missing defence and goalkeeper (and vice versa).
    """
    features = features.assign(_day=features["date"].dt.tz_convert("Europe/London").dt.date.astype(str)).drop_duplicates(["team", "_day"])
    lookup = features.set_index(["team", "_day"])[list(FEATURES)]
    out = matches.copy()
    day = pd.to_datetime(out["kickoff_utc"], utc=True).dt.tz_convert("Europe/London").dt.date.astype(str)
    sides = {}
    for side in ("home", "away"):
        index = pd.MultiIndex.from_arrays([out[side], day])
        sides[side] = lookup.reindex(index).reset_index(drop=True).fillna(0.0)
    for side, other in (("home", "away"), ("away", "home")):
        out[f"{side}_att_missing"] = sides[side]["att_missing"].to_numpy()
        out[f"{side}_opp_def_missing"] = sides[other]["def_missing"].to_numpy()
        out[f"{side}_opp_gk_missing"] = sides[other]["gk_missing"].to_numpy()
    return out


COVARIATES = ("att_missing", "opp_def_missing", "opp_gk_missing")
