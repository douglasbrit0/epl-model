"""Persist canonical match tables as parquet."""

from pathlib import Path

import pandas as pd


def save_matches(matches: pd.DataFrame, path: Path) -> Path:
    """Write processed match data to parquet, requiring the canonical core schema."""
    required = {"fixture_id", "kickoff_utc", "home", "away", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Processed matches missing columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(path, index=False)
    return path


def load_matches(path: Path) -> pd.DataFrame:
    """Load processed match parquet and verify it is readable."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
