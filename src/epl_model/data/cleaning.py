"""Canonical joins between results and xG records."""

import logging
from pathlib import Path

import pandas as pd

from epl_model.data.names import normalize_match_teams

LOGGER = logging.getLogger(__name__)


def join_xg(results: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """Join normalized team xG by exact UTC date and teams; unmatched games fail."""
    result = normalize_match_teams(results, "football_data")
    expected = normalize_match_teams(xg, "understat")
    for frame in (result, expected):
        frame["match_date"] = pd.to_datetime(frame["kickoff_utc"], utc=True).dt.date
    key = ["match_date", "home", "away"]
    joined = result.merge(
        expected[key + ["npxg_home", "npxg_away"]],
        how="left",
        on=key,
        validate="one_to_one",
        indicator=True,
    )
    missing = joined.loc[joined["_merge"] != "both", key]
    if not missing.empty:
        sample = missing.head(5).to_dict("records")
        raise AssertionError(f"Results without exact Understat xG match: {len(missing)}; examples={sample}")
    return joined.drop(columns=["_merge", "match_date"])


def attach_optional_xg(results: pd.DataFrame, xg_path: Path) -> pd.DataFrame:
    """Join an npxG export when one is supplied; otherwise run goals-only with a warning."""
    if not xg_path.exists():
        LOGGER.warning("No xG export at %s; running the goals-only model", xg_path)
        return results
    from epl_model.data.understat import load_understat_npxg

    return join_xg(results, load_understat_npxg(xg_path))
