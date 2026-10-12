"""Reproduce walk-forward metrics, market benchmark, calibration and P&L reports."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epl_model.data.cleaning import attach_optional_xg
from epl_model.data.football_data import download_history, load_history
from epl_model.evaluation.backtest import flat_stake_edges, flat_stake_profit, summarize_walk_forward
from epl_model.evaluation.metrics import closing_odds_probabilities, reliability_bins
from epl_model.data.names import normalize_match_teams
from epl_model.evaluation.tuning import nested_select, run_grid, save_tuning
from epl_model.pipeline import backtest_summary

ROOT = Path(__file__).resolve().parent


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    history = load_history(download_history(ROOT / "data" / "raw"))
    history = normalize_match_teams(history, "football_data")
    matches = attach_optional_xg(history, ROOT / "data" / "raw" / "understat_npxg.csv")
    # Every grid configuration is backtested; each season uses the one that was best on earlier seasons.
    grid = run_grid(matches)
    baseline, chosen, production = nested_select(grid)
    save_tuning(reports / "tuning.json", chosen, production)
    print(f"Production config: {production}")
    baseline["flat_stake_profit"] = flat_stake_profit(baseline, threshold=0.02)
    baseline.to_csv(reports / "walk_forward.csv", index=False)
    (reports / "backtest_summary.json").write_text(json.dumps(backtest_summary(baseline), indent=2) + "\n", encoding="utf-8")
    summary = summarize_walk_forward(baseline)
    pd.DataFrame([summary]).to_csv(reports / "metrics.csv", index=False)
    edges = flat_stake_edges(baseline)
    edges.to_csv(reports / "flat_stake_edges.csv", index=False)
    calibration = reliability_bins(baseline[["p_home", "p_draw", "p_away"]].to_numpy(), baseline["outcome"].to_numpy(dtype=int))
    pd.DataFrame(calibration).to_csv(reports / "calibration_bins.csv", index=False)
    probabilities = baseline[["p_home", "p_draw", "p_away"]].to_numpy()
    observed = np.eye(3)[baseline["outcome"].to_numpy(dtype=int)]
    edges_x = np.linspace(0, 1, 11)
    plt.figure(figsize=(7, 6))
    for index, label in enumerate(("Home", "Draw", "Away")):
        bins = []
        for left, right in zip(edges_x[:-1], edges_x[1:], strict=True):
            mask = (probabilities[:, index] >= left) & (probabilities[:, index] < right)
            if mask.any():
                bins.append((probabilities[mask, index].mean(), observed[mask, index].mean()))
        if bins:
            plt.plot(*np.array(bins).T, marker="o", label=label)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports / "reliability.png", dpi=160)
    plt.close()
    print(pd.DataFrame([summary]).to_string(index=False))
    print("Pinnacle/market benchmark and Shin de-vigging are available in the metrics module.")


if __name__ == "__main__":
    main()
