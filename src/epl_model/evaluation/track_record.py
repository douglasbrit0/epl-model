"""Score previously logged forecasts when completed results arrive."""

import json
from pathlib import Path

import pandas as pd

from epl_model.evaluation.metrics import closing_odds_probabilities, multiclass_scores


def score_prediction_log(log_path: Path, results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Append a scored record for each newly finished logged prediction."""
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    logged = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    predictions = pd.DataFrame(logged)
    if not {"home", "away", "kickoff_utc"}.issubset(predictions.columns):
        raise ValueError("Prediction log must include home, away, and kickoff_utc for result matching")
    predictions["match_date"] = pd.to_datetime(predictions["kickoff_utc"], utc=True).dt.date
    result_rows = results.copy()
    result_rows["match_date"] = pd.to_datetime(result_rows["kickoff_utc"], utc=True).dt.date
    result_columns = ["match_date", "home", "away", "home_goals", "away_goals", "odds_home", "odds_draw", "odds_away"]
    result_columns = [column for column in result_columns if column in result_rows]
    merged = predictions.merge(result_rows[result_columns], on=["match_date", "home", "away"], how="inner", validate="many_to_one")
    if merged.empty:
        return merged
    already = set()
    if output_path.exists():
        already = {json.loads(line)["prediction_id"] for line in output_path.read_text().splitlines() if line.strip()}
    additions = []
    for row in merged.to_dict("records"):
        prediction_id = row["prediction_id"]
        if prediction_id in already:
            continue
        actual = 0 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 2
        probs = [row["p_home"], row["p_draw"], row["p_away"]]
        score = multiclass_scores([probs], [actual])
        record = {"prediction_id": prediction_id, "fixture_id": row["fixture_id"], "actual_outcome": actual, **{f"model_{key}": value for key, value in score.items()}}
        odds = [row.get("odds_home"), row.get("odds_draw"), row.get("odds_away")]
        if all(value is not None and pd.notna(value) and float(value) > 1 for value in odds):
            market = closing_odds_probabilities(tuple(float(value) for value in odds), "power")
            record.update({f"market_{key}": value for key, value in multiclass_scores([market], [actual]).items()})
            shin = closing_odds_probabilities(tuple(float(value) for value in odds), "shin")
            record.update({f"market_shin_{key}": value for key, value in multiclass_scores([shin], [actual]).items()})
            model_probabilities = [row["p_home"], row["p_draw"], row["p_away"]]
            profit = 0.0
            for index, model_probability in enumerate(model_probabilities):
                if model_probability - market[index] >= 0.02:
                    profit += float(odds[index]) - 1 if actual == index else -1
            record["flat_stake_profit"] = profit
        additions.append(record)
        already.add(prediction_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        for row in additions:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return pd.DataFrame(additions)
