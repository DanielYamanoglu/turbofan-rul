"""Train, cross-validate by engine, and score on the official test split."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data, metrics
from .features import build_features, feature_names, informative_sensors

MODEL_DIR = Path(os.environ.get("TURBOFAN_MODEL_DIR", data.ROOT / "models"))
PARAMS = {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0, "random_state": 0}
FOLDS = 5


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**PARAMS)


def predict(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """RUL for every cycle in ``frame``; clipped to the range the model was trained on."""
    feats = build_features(frame, bundle["sensors"])
    feats["rul"] = np.clip(bundle["model"].predict(feats[bundle["features"]]), 0, data.RUL_CAP)
    return feats[["unit", "cycle", "rul"]]


def last_cycle(predictions: pd.DataFrame) -> pd.Series:
    return predictions.sort_values("cycle").groupby("unit")["rul"].last()


def run(subset: str = "FD001", model_dir: Path = MODEL_DIR) -> dict:
    train = data.load_train(subset).sort_values(["unit", "cycle"]).reset_index(drop=True)
    sensors = informative_sensors(train, data.SENSORS)
    names = feature_names(sensors)
    X = build_features(train, sensors)[names]
    y, groups = train["rul"], train["unit"]

    # Folds are split by engine: cycles of one engine never sit on both sides.
    cv = GroupKFold(n_splits=FOLDS)
    baseline = make_pipeline(StandardScaler(), Ridge())
    cv_baseline = cross_val_predict(baseline, train[sensors], y, groups=groups, cv=cv)
    cv_model = cross_val_predict(make_model(), X, y, groups=groups, cv=cv)

    bundle = {"model": make_model().fit(X, y), "sensors": sensors, "features": names, "subset": subset}
    baseline.fit(train[sensors], y)

    test, raw_truth = data.load_test(subset)
    truth = raw_truth.clip(upper=data.RUL_CAP)
    predicted = last_cycle(predict(bundle, test)).loc[truth.index]
    last_rows = test.sort_values("cycle").groupby("unit").last().loc[truth.index]
    baseline_predicted = np.clip(baseline.predict(last_rows[sensors]), 0, data.RUL_CAP)

    report = {
        "subset": subset,
        "train_engines": int(groups.nunique()),
        "train_rows": int(len(train)),
        "test_engines": int(len(truth)),
        "features": len(names),
        "cv_rmse": {"ridge_raw_sensors": metrics.rmse(y, cv_baseline), "gradient_boosting": metrics.rmse(y, cv_model)},
        "test": {
            "ridge_raw_sensors": {
                "rmse": metrics.rmse(truth, baseline_predicted),
                "nasa_score": metrics.nasa_score(truth, baseline_predicted),
            },
            "gradient_boosting": {
                "rmse": metrics.rmse(truth, predicted),
                "nasa_score": metrics.nasa_score(truth, predicted),
                # Same predictions against the uncapped labels, for comparison with papers that do not cap.
                "rmse_uncapped_labels": metrics.rmse(raw_truth, predicted),
                "alert": metrics.alert_metrics(truth, predicted),
            },
        },
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model.joblib", compress=3)
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    _log_to_mlflow(report)
    return report


def _log_to_mlflow(report: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{data.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("turbofan-rul")
    with mlflow.start_run():
        mlflow.log_params({**PARAMS, "subset": report["subset"], "features": report["features"], "folds": FOLDS})
        boosted = report["test"]["gradient_boosting"]
        mlflow.log_metrics(
            {
                "cv_rmse": report["cv_rmse"]["gradient_boosting"],
                "test_rmse": boosted["rmse"],
                "test_nasa_score": boosted["nasa_score"],
                "alert_precision": boosted["alert"]["precision"],
                "alert_recall": boosted["alert"]["recall"],
            }
        )
        mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))
