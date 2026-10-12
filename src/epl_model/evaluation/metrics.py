"""Probabilistic 1X2 scoring and reliability summaries."""

import numpy as np
import pandas as pd


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    """Aggregate scoreline probabilities in home/draw/away order."""
    home = draw = away = 0.0
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            value = matrix[home_goals, away_goals]
            if home_goals > away_goals:
                home += value
            elif home_goals == away_goals:
                draw += value
            else:
                away += value
    probs = np.array([home, draw, away], dtype=float)
    return probs / probs.sum()


def score_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Derive match markets from a scoreline grid."""
    probs = outcome_probabilities(matrix)
    total = np.indices(matrix.shape).sum(axis=0)
    return {
        "p_home": float(probs[0]),
        "p_draw": float(probs[1]),
        "p_away": float(probs[2]),
        "p_over_2_5": float(matrix[total >= 3].sum()),
        "p_btts": float(matrix[1:, 1:].sum()),
    }


def multiclass_scores(probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """Calculate multiclass log-loss, Brier score, and ranked probability score."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1)
    outcomes = np.asarray(outcomes, dtype=int)
    one_hot = np.eye(3)[outcomes]
    log_loss = -np.log(probabilities[np.arange(len(outcomes)), outcomes]).mean()
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    cumulative_p = np.cumsum(probabilities, axis=1)[:, :-1]
    cumulative_y = np.cumsum(one_hot, axis=1)[:, :-1]
    rps = np.square(cumulative_p - cumulative_y).mean()
    return {"log_loss": float(log_loss), "brier": float(brier), "rps": float(rps)}


def reliability_bins(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    """Return one-vs-rest calibration bins pooled over the three outcomes."""
    probs = np.asarray(probabilities).reshape(-1)
    observed = np.eye(3)[np.asarray(outcomes, dtype=int)].reshape(-1)
    edges = np.linspace(0, 1, bins + 1)
    records = []
    for index in range(bins):
        mask = (probs >= edges[index]) & (probs < edges[index + 1] if index < bins - 1 else probs <= 1)
        if mask.any():
            records.append(
                {"bin": index, "mean_probability": float(probs[mask].mean()), "observed_frequency": float(observed[mask].mean()), "count": int(mask.sum())}
            )
    return records


def closing_odds_probabilities(odds: tuple[float, float, float], method: str = "power") -> np.ndarray:
    """De-vig decimal 1X2 prices with power or Shin's insider-trading method."""
    from scipy.optimize import brentq

    inverse = 1 / np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(inverse)) or np.any(inverse <= 0):
        raise ValueError("Closing odds must be finite positive decimal prices")
    if method == "power":
        exponent = brentq(lambda power: np.sum(inverse**power) - 1, 0.01, 20)
        return inverse**exponent
    if method == "shin":
        total = inverse.sum()

        def probabilities(z: float) -> np.ndarray:
            return (np.sqrt(z * z + 4 * (1 - z) * inverse * inverse / total) - z) / (2 * (1 - z))

        try:
            z = brentq(lambda value: probabilities(value).sum() - 1, 0, 0.999)
        except ValueError:
            z = 0.0
        result = probabilities(z)
        return result / result.sum()
    raise ValueError("method must be 'power' or 'shin'")
