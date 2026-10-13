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

    def broken(sport, key):
        raise ConnectionError("down")

    monkeypatch.setattr(odds_api, "fetch_odds", broken)
    assert odds_api.take_odds_snapshot(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_snapshot_writes_one_file_per_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "x")
    monkeypatch.setattr(odds_api, "fetch_odds", lambda sport, key: ([EVENT], "492"))
    path = odds_api.take_odds_snapshot(tmp_path, now=datetime(2026, 10, 13, 6, tzinfo=timezone.utc))
    loaded = odds_api.load_odds_snapshots(tmp_path)
    assert path.name == "odds_20261013T060000Z.csv.gz"
    assert set(loaded["sport"]) == {"soccer_epl", "soccer_efl_champ"}
