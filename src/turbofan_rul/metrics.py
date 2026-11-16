from __future__ import annotations

import numpy as np

ALERT_HORIZON = 30  # cycles: "this engine needs maintenance soon"


def rmse(true, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(true)) ** 2)))


def nasa_score(true, predicted) -> float:
    """PHM08 scoring function: predicting too late is penalised harder than too early."""
    error = np.asarray(predicted, dtype=float) - np.asarray(true, dtype=float)
    return float(np.sum(np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)))


def alert_metrics(true, predicted, horizon: int = ALERT_HORIZON) -> dict:
    actual = np.asarray(true) <= horizon
    flagged = np.asarray(predicted) <= horizon
    tp = int(np.sum(actual & flagged))
    return {
        "horizon": horizon,
        "engines_near_failure": int(actual.sum()),
        "precision": tp / int(flagged.sum()) if flagged.any() else 0.0,
        "recall": tp / int(actual.sum()) if actual.any() else 0.0,
    }


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index; above 0.2 is the conventional 'significant shift' mark."""
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))
