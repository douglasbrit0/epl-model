"""Daily timestamped bookmaker-price snapshots from The Odds API (the-odds-api.com, v4).

football-data.co.uk only records one pre-match and one closing price per bookmaker, which is not
enough to test betting at a chosen time or line shopping against Pinnacle. Each run stores every
bookmaker's 1X2 and totals prices for upcoming matches, with that bookmaker's own last-update time.

Requires ODDS_API_KEY (free tier: 500 credits/month). Each sport costs one credit per market per
region, so the default 2 sports x 2 markets x 2 regions = 8 credits per run. Team names are stored as
the API spells them; mapping to canonical names happens when the snapshots are used.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORTS = ("soccer_epl", "soccer_efl_champ")
MARKETS = "h2h,totals"
# "eu" includes Pinnacle, the sharp reference price; "uk" covers the bookmakers available to bet with.
REGIONS = "uk,eu"
COLUMNS = ["fetched_at", "sport", "event_id", "commence_time", "home", "away", "bookmaker", "bookmaker_update",
           "market", "outcome", "point", "price"]


def fetch_odds(sport: str, api_key: str, timeout: int = 30) -> tuple[list, str | None]:
    """Fetch current prices for one sport; also returns the remaining-credit header."""
    params = {"apiKey": api_key, "regions": REGIONS, "markets": MARKETS, "oddsFormat": "decimal", "dateFormat": "iso"}
    response = requests.get(API_URL.format(sport=sport), params=params, timeout=timeout)
    response.raise_for_status()
    events = response.json()
    if not isinstance(events, list):
        raise ValueError(f"Unexpected Odds API response for {sport}")
    return events, response.headers.get("x-requests-remaining")


def flatten(events: list, sport: str, fetched_at: datetime) -> pd.DataFrame:
    """One row per event x bookmaker x market x outcome."""
    stamp = pd.Timestamp(fetched_at).tz_convert("UTC").isoformat().replace("+00:00", "Z")
    rows = []
    for event in events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "fetched_at": stamp,
                        "sport": sport,
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "home": event.get("home_team"),
                        "away": event.get("away_team"),
                        "bookmaker": bookmaker.get("key"),
                        "bookmaker_update": market.get("last_update") or bookmaker.get("last_update"),
                        "market": market.get("key"),
                        "outcome": outcome.get("name"),
                        "point": outcome.get("point"),
                        "price": outcome.get("price"),
                    })
    return pd.DataFrame(rows, columns=COLUMNS)


def take_odds_snapshot(directory: Path, now: datetime | None = None, sports: tuple[str, ...] = SPORTS) -> Path | None:
    """Fetch and store one snapshot; a missing key or any failure is logged and never raised."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        LOGGER.info("ODDS_API_KEY not set; skipping odds snapshot")
        return None
    fetched_at = now or datetime.now(timezone.utc)
    frames = []
    for sport in sports:
        try:
            events, remaining = fetch_odds(sport, api_key)
        except Exception as error:  # noqa: BLE001 - odds data must never block predictions
            LOGGER.warning("Odds snapshot for %s failed: %s", sport, error)
            continue
        frames.append(flatten(events, sport, fetched_at))
        LOGGER.info("Odds snapshot %s: %d events (credits remaining: %s)", sport, len(events), remaining)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"odds_{frame['fetched_at'].iloc[0].replace(':', '').replace('-', '')}.csv.gz"
    if path.exists():
        LOGGER.warning("Odds snapshot %s already exists; not overwritten", path.name)
        return None
    frame.to_csv(path, index=False, compression="gzip")
    return path


def load_odds_snapshots(directory: Path) -> pd.DataFrame:
    """Load every stored odds snapshot (empty frame if none)."""
    paths = sorted(directory.glob("odds_*.csv.gz"))
    if not paths:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
