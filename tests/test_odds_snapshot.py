from datetime import datetime, timezone

from epl_model.data import odds_api

EVENT = {
    "id": "abc", "commence_time": "2026-10-17T14:00:00Z", "home_team": "Fulham", "away_team": "Hull City",
    "bookmakers": [{
        "key": "pinnacle", "last_update": "2026-10-13T05:58:00Z",
        "markets": [
            {"key": "h2h", "last_update": "2026-10-13T05:58:00Z", "outcomes": [
                {"name": "Fulham", "price": 2.05}, {"name": "Hull City", "price": 3.9}, {"name": "Draw", "price": 3.5}]},
            {"key": "totals", "last_update": "2026-10-13T05:57:00Z", "outcomes": [
                {"name": "Over", "price": 1.95, "point": 2.5}, {"name": "Under", "price": 1.9, "point": 2.5}]},
        ],
    }],
}


def test_flatten_keeps_every_price_with_bookmaker_timestamps():
    frame = odds_api.flatten([EVENT], "soccer_epl", datetime(2026, 10, 13, 6, tzinfo=timezone.utc))
    assert len(frame) == 5
    assert frame["fetched_at"].unique().tolist() == ["2026-10-13T06:00:00Z"]
    totals = frame.loc[frame["market"] == "totals"]
    assert totals["point"].tolist() == [2.5, 2.5]
    assert totals["bookmaker_update"].unique().tolist() == ["2026-10-13T05:57:00Z"]


def test_snapshot_skips_without_key_and_survives_api_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert odds_api.take_odds_snapshot(tmp_path) is None
    monkeypatch.setenv("ODDS_API_KEY", "x")

    def broken(*args):
        raise ConnectionError("down")

    monkeypatch.setattr(odds_api, "fetch_odds", broken)
    monkeypatch.setattr(odds_api, "outright_sports", broken)
    assert odds_api.take_odds_snapshot(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_snapshot_writes_one_file_per_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "x")
    calls = []

    def fake(sport, key, markets, regions):
        calls.append((sport, markets, regions))
        return [EVENT], "492"

    monkeypatch.setattr(odds_api, "fetch_odds", fake)
    monkeypatch.setattr(odds_api, "outright_sports", lambda key: ["soccer_epl_winner"])
    path = odds_api.take_odds_snapshot(tmp_path, now=datetime(2026, 10, 13, 6, tzinfo=timezone.utc))
    loaded = odds_api.load_odds_snapshots(tmp_path)
    assert path.name == "odds_20261013T060000Z.csv.gz"
    assert set(loaded["sport"]) == {"soccer_epl", "soccer_efl_champ", "soccer_epl_winner"}
    assert ("soccer_epl", "h2h,totals", "uk,eu,us_ex") in calls
    assert ("soccer_epl_winner", "outrights", "eu,us_ex") in calls


def test_market_odds_attach_pinnacle_prices_by_teams_and_date(tmp_path, monkeypatch):
    import pandas as pd

    soft = {"key": "skybet", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Brighton and Hove Albion", "price": 2.2}, {"name": "Arsenal", "price": 3.1}, {"name": "Draw", "price": 3.4}]}]}
    pinnacle = {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Brighton and Hove Albion", "price": 2.1}, {"name": "Arsenal", "price": 3.3}, {"name": "Draw", "price": 3.5}]}]}
    event = {"id": "e1", "commence_time": "2026-10-18T13:00:00Z", "home_team": "Brighton and Hove Albion", "away_team": "Arsenal",
             "bookmakers": [soft, pinnacle]}
    monkeypatch.setenv("ODDS_API_KEY", "x")
    monkeypatch.setattr(odds_api, "fetch_odds", lambda sport, key, markets, regions: ([event], "490"))
    path = odds_api.take_odds_snapshot(tmp_path, requests_=(("soccer_epl", "h2h", "eu"),), outrights=False)
    fixtures = pd.DataFrame([
        {"fixture_id": "fdo:1", "kickoff_utc": "2026-10-18T13:00:00Z", "home": "Brighton", "away": "Arsenal"},
        {"fixture_id": "fdo:2", "kickoff_utc": "2026-10-18T15:00:00Z", "home": "Chelsea", "away": "Fulham"},
    ])
    out = odds_api.attach_market_odds(fixtures, path)
    assert out.loc[0, ["odds_home", "odds_draw", "odds_away"]].tolist() == [2.1, 3.5, 3.3]
    assert out.loc[0, "odds_source"] == "pinnacle"
    assert pd.isna(out.loc[1, "odds_home"])
