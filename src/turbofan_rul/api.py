"""HTTP service: send an engine's sensor history, get its remaining useful life."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from . import data
from .metrics import ALERT_HORIZON
from .train import MODEL_DIR, last_cycle, predict

Reading = create_model(
    "Reading",
    cycle=(int, Field(ge=1)),
    **{name: (float, ...) for name in (*data.SETTINGS, *data.SENSORS)},
)


class PredictRequest(BaseModel):
    history: list[Reading] = Field(min_length=1, description="All cycles observed so far for one engine")


class PredictResponse(BaseModel):
    rul_cycles: float
    maintenance_alert: bool
    cycles_seen: int


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(MODEL_DIR / "model.joblib")


app = FastAPI(title="turbofan-rul")


@app.get("/health")
def health() -> dict:
    """Liveness: the process answers. Kept free of I/O so a slow disk never gets the pod restarted."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the model bundle loads, so traffic only reaches pods that can actually predict."""
    load_bundle()
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict_rul(request: PredictRequest) -> PredictResponse:
    frame = pd.DataFrame([reading.model_dump() for reading in request.history]).assign(unit=1)
    rul = float(last_cycle(predict(load_bundle(), frame)).iloc[0])
    return PredictResponse(rul_cycles=round(rul, 1), maintenance_alert=rul <= ALERT_HORIZON, cycles_seen=len(frame))
