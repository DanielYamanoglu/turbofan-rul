"""Causal time-series features, written as SQL window functions and run in DuckDB."""
from __future__ import annotations

from typing import Sequence

import duckdb
import pandas as pd

SHORT_WINDOW = 10
LONG_WINDOW = 30
BASELINE_CYCLES = 20  # an engine's first cycles define its own healthy reference
KEYS = ["unit", "cycle"]


def feature_sql(sensors: Sequence[str]) -> str:
    """Every window ends at the current row, so a feature never sees the future."""
    past = "PARTITION BY unit ORDER BY cycle ROWS BETWEEN {n} PRECEDING AND CURRENT ROW"
    short, long_ = past.format(n=SHORT_WINDOW - 1), past.format(n=LONG_WINDOW - 1)
    history = "PARTITION BY unit ORDER BY cycle ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    columns = ["unit", "cycle"]
    for s in sensors:
        columns += [
            f"{s}",
            f"AVG({s}) OVER ({short}) AS {s}_mean{SHORT_WINDOW}",
            f"AVG({s}) OVER ({long_}) AS {s}_mean{LONG_WINDOW}",
            f"COALESCE(STDDEV_SAMP({s}) OVER ({long_}), 0) AS {s}_std{LONG_WINDOW}",
            f"AVG({s}) OVER ({short}) - AVG(CASE WHEN cycle <= {BASELINE_CYCLES} THEN {s} END) OVER ({history})"
            f" AS {s}_drift",
        ]
    return "SELECT " + ",\n       ".join(columns) + "\nFROM trajectories\nORDER BY unit, cycle"


def build_features(frame: pd.DataFrame, sensors: Sequence[str]) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.register("trajectories", frame)
        return connection.execute(feature_sql(sensors)).df()
    finally:
        connection.close()


def feature_names(sensors: Sequence[str]) -> list[str]:
    names = ["cycle"]
    for s in sensors:
        names += [s, f"{s}_mean{SHORT_WINDOW}", f"{s}_mean{LONG_WINDOW}", f"{s}_std{LONG_WINDOW}", f"{s}_drift"]
    return names


def informative_sensors(frame: pd.DataFrame, sensors: Sequence[str]) -> list[str]:
    """Sensors that never move carry no degradation signal."""
    return [s for s in sensors if frame[s].std() > 1e-6]
