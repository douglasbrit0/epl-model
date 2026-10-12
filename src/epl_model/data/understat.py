"""Strict Understat non-penalty-xG import boundary.

Understat's match/shot pages were not verifiable from the build environment. This
adapter intentionally accepts an explicit Understat export rather than relabelling
total xG as non-penalty xG or silently substituting a different provider.
"""

from pathlib import Path

import pandas as pd


REQUIRED = {"kickoff_utc", "home", "away", "npxg_home", "npxg_away"}


def load_understat_npxg(path: Path) -> pd.DataFrame:
    """Load a team-match export and require explicit non-penalty xG columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Understat export not found at {path}; provide match-level npxg_home/npxg_away data. "
            "Total xG is not accepted as a substitute."
        )
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Understat export lacks required non-penalty xG fields: {sorted(missing)}")
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="raise")
    for field in ("npxg_home", "npxg_away"):
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if (frame[field] < 0).any():
            raise ValueError(f"Understat field {field} cannot be negative")
    return frame
