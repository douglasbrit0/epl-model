"""Daily timestamped bookmaker-price snapshots from The Odds API (the-odds-api.com, v4).

football-data.co.uk only records one pre-match and one closing price per bookmaker, which is not
enough to test betting at a chosen time or line shopping against Pinnacle. Each run stores every
bookmaker's 1X2 and totals prices for upcoming matches, with that bookmaker's own last-update time.

Requires ODDS_API_KEY (free tier: 500 credits/month). Each request costs one credit per market per
region. Defaults, about 12 credits per run:
- EPL matches: 1X2 + totals from UK and EU bookmakers (EU includes Pinnacle, the sharp reference)
  and US exchanges (``us_ex``: Kalshi, Polymarket, ...), the venues usable from Texas.
- Championship matches: 1X2 + totals, UK and EU (research only).
- EPL outright (season) markets found via the free sports list: EU and US exchanges.
Team names are stored as the API spells them; mapping happens when snapshots are used.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORTS_URL = "https://api.the-odds-api.com/v4/sports"
# (sport, markets, regions) per request.
REQUESTS = (
    ("soccer_epl", "h2h,totals", "uk,eu,us_ex"),
    ("soccer_efl_champ", "h2h,totals", "uk,eu"),
)
OUTRIGHT_PREFIX, OUTRIGHT_REGIONS = "soccer_epl", "eu,us_ex"
COLUMNS = ["fetched_at", "sport", "event_id", "commence_time", "home", "away", "bookmaker", "bookmaker_update",
           "market", "outcome", "point", "price"]


def outright_sports(api_key: str, timeout: int = 30) -> list[str]:
    """Active EPL outright (season) markets; listing sports costs no credits."""
    response = requests.get(SPORTS_URL, params={"apiKey": api_key}, timeout=timeout)
    response.raise_for_status()
    return [sport["key"] for sport in response.json() if sport.get("has_outrights") and sport.get("key", "").startswith(OUTRIGHT_PREFIX)]


def fetch_odds(sport: str, api_key: str, markets: str = "h2h,totals", regions: str = "uk,eu", timeout: int = 30) -> tuple[list, str | None]:
    """Fetch current prices for one sport; also returns the remaining-credit header."""
    params = {"apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": "decimal", "dateFormat": "iso"}
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


def take_odds_snapshot(directory: Path, now: datetime | None = None, requests_: tuple[tuple[str, str, str], ...] = REQUESTS, outrights: bool = True) -> Path | None:
    """Fetch and store one snapshot; a missing key or any failure is logged and never raised."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        LOGGER.info("ODDS_API_KEY not set; skipping odds snapshot")
        return None
    fetched_at = now or datetime.now(timezone.utc)
    plan = list(requests_)
    if outrights:
        try:
            plan += [(sport, "outrights", OUTRIGHT_REGIONS) for sport in outright_sports(api_key)]
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Could not list outright markets: %s", error)
    frames = []
    for sport, markets, regions in plan:
        try:
            events, remaining = fetch_odds(sport, api_key, markets, regions)
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


def match_odds(snapshot: pd.DataFrame, sport: str = "soccer_epl") -> pd.DataFrame:
    """1X2 prices per match from one snapshot: Pinnacle where quoted, else the bookmaker median."""
    from epl_model.data.names import normalize_team_names

    h2h = snapshot.loc[(snapshot["sport"] == sport) & (snapshot["market"] == "h2h")].copy()
    if h2h.empty:
        return pd.DataFrame(columns=["home", "away", "commence_time", "odds_home", "odds_draw", "odds_away", "odds_source"])
    h2h["side"] = np.select([h2h["outcome"] == h2h["home"], h2h["outcome"] == h2h["away"]], ["odds_home", "odds_away"], "odds_draw")
    keys = ["event_id", "home", "away", "commence_time"]
    wide = h2h.pivot_table(index=keys + ["bookmaker"], columns="side", values="price").dropna().reset_index()
    rows = []
    for _, group in wide.groupby(keys):
        pinnacle = group.loc[group["bookmaker"] == "pinnacle"]
        prices = ["odds_home", "odds_draw", "odds_away"]
        chosen, source = (pinnacle.iloc[0], "pinnacle") if not pinnacle.empty else (group[prices].median(), "bookmaker_median")
        first = group.iloc[0]
        rows.append({"home": first["home"], "away": first["away"], "commence_time": first["commence_time"],
                     **{column: float(chosen[column]) for column in prices}, "odds_source": source})
    prices = pd.DataFrame(rows)
    prices["home"] = normalize_team_names(prices["home"], "odds_api")
    prices["away"] = normalize_team_names(prices["away"], "odds_api")
    return prices


def attach_market_odds(fixtures: pd.DataFrame, snapshot_path: Path | None) -> pd.DataFrame:
    """Add odds_home/draw/away and odds_source to fixtures (same teams, kickoff within a day).

    Any problem leaves the fixtures without market fields and logs a warning.
    """
    if snapshot_path is None or fixtures.empty:
        return fixtures
    try:
        prices = match_odds(pd.read_csv(snapshot_path))
        merged = fixtures.merge(prices, on=["home", "away"], how="left", validate="many_to_one")
        gap = (pd.to_datetime(merged["kickoff_utc"], utc=True) - pd.to_datetime(merged["commence_time"], utc=True)).abs()
        stale = gap > pd.Timedelta(days=1)
        merged.loc[stale, ["odds_home", "odds_draw", "odds_away", "odds_source"]] = None
        LOGGER.info("Attached market odds to %d of %d fixtures", int(merged["odds_home"].notna().sum()), len(merged))
        return merged.drop(columns="commence_time")
    except Exception as error:  # noqa: BLE001 - market fields are optional
        LOGGER.warning("Could not attach market odds: %s", error)
        return fixtures
