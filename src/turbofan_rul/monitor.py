"""Input drift check: compare incoming sensor distributions with the training data."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .metrics import psi

PSI_ALERT = 0.2


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: Sequence[str]) -> dict:
    scores = {column: psi(reference[column], current[column]) for column in columns}
    drifted = sorted((c for c, value in scores.items() if value > PSI_ALERT), key=lambda c: -scores[c])
    return {"checked": len(scores), "drifted": drifted, "max_psi": max(scores.values()), "psi": scores}
