"""Football-data.co.uk historical E0 results and closing odds ingestion."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
PINNACLE = ("PSCH", "PSCD", "PSCA")
MARKET_AVERAGE = ("AvgCH", "AvgCD", "AvgCA")
LOGGER = logging.getLogger(__name__)


def season_codes(first_start_year: int = 2017, last_start_year: int | None = None) -> list[str]:
    """Return football-data.co.uk season directory names from 2017/18 onward."""
    last = last_start_year or pd.Timestamp.now(tz="UTC").year
    return [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(first_start_year, last + 1)]


def download_history(raw_dir: Path, first_start_year: int = 2017, last_start_year: int | None = None, refresh_seasons: int = 2) -> list[Path]:
    """Download EPL CSVs; completed seasons are fetched once, the latest ones every run.

    If refreshing a season fails but a cached copy exists, the cached copy is used with a
    warning, so a transient outage does not stop the pipeline.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    codes = season_codes(first_start_year, last_start_year)
    paths = []
    for position, season in enumerate(codes):
        target = raw_dir / f"E0_{season}.csv"
        refresh = position >= len(codes) - refresh_seasons
        if target.exists() and not refresh:
            paths.append(target)
            continue
        try:
            response = requests.get(BASE_URL.format(season=season), timeout=45)
            response.raise_for_status()
            if not response.content.strip():
                raise RuntimeError(f"Empty football-data.co.uk response for season {season}")
        except (requests.RequestException, RuntimeError) as error:
            if not target.exists():
                raise
            LOGGER.warning("Could not refresh %s (%s); using cached copy", target.name, error)
            paths.append(target)
            continue
        target.write_bytes(response.content)
        paths.append(target)
    if not paths:
        raise RuntimeError("No football-data.co.uk season files downloaded")
    return paths


def _closing_odds(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer Pinnacle closing prices; fall back only to the published market average."""
    out = frame.copy()
    for side, pinnacle, average in zip(("home", "draw", "away"), PINNACLE, MARKET_AVERAGE, strict=True):
        primary = pd.to_numeric(out[pinnacle], errors="coerce") if pinnacle in out else pd.Series(index=out.index, dtype=float)
        fallback = pd.to_numeric(out[average], errors="coerce") if average in out else pd.Series(index=out.index, dtype=float)
        out[f"odds_{side}"] = primary.where(primary.gt(1), fallback.where(fallback.gt(1)))
    out["odds_source"] = "pinnacle"
    average_used = out[list(f"odds_{side}" for side in ("home", "draw", "away"))].notna().all(axis=1)
    pinnacle_available = pd.Series(True, index=out.index)
    for col in PINNACLE:
        pinnacle_available &= pd.to_numeric(out[col], errors="coerce").gt(1) if col in out else False
    out.loc[average_used & ~pinnacle_available, "odds_source"] = "market_average"
    return out


def load_history(paths: list[Path]) -> pd.DataFrame:
    """Read source CSVs, harmonize core fields, retain closing-odds provenance."""
    frames = []
    for path in paths:
        raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{path} missing football-data columns: {sorted(missing)}")
        raw = _closing_odds(raw)
        date = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", utc=True)
        part = pd.DataFrame(
            {
                "kickoff_utc": date,
                "home": raw["HomeTeam"].astype(str).str.strip(),
                "away": raw["AwayTeam"].astype(str).str.strip(),
                "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
                "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
                "odds_home": raw["odds_home"],
                "odds_draw": raw["odds_draw"],
                "odds_away": raw["odds_away"],
                "odds_source": raw["odds_source"],
            }
        )
        part = part.dropna(subset=["kickoff_utc", "home_goals", "away_goals"])
        part["fixture_id"] = part.apply(
            lambda row: f"fd:{row.kickoff_utc.date()}:{row.home}:{row.away}", axis=1
        )
        frames.append(part)
    if not frames:
        raise ValueError("No historical CSV paths were provided")
    result = pd.concat(frames, ignore_index=True).drop_duplicates("fixture_id")
    return result.sort_values("kickoff_utc").reset_index(drop=True)
