import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from turbofan_rul import api, data, metrics
from turbofan_rul.features import build_features, feature_names
from turbofan_rul.monitor import drift_report


def _frame(values, unit=1):
    rows = [{"unit": unit, "cycle": i + 1, "s2": float(v)} for i, v in enumerate(values)]
    return pd.DataFrame(rows)


def test_rul_counts_down_to_failure_and_is_capped():
    frame = pd.DataFrame({"unit": [1] * 200, "cycle": range(1, 201)})
    rul = data.add_rul(frame)["rul"]
    assert rul.iloc[-1] == 0 and rul.iloc[-2] == 1 and rul.iloc[0] == data.RUL_CAP


def test_rolling_features_are_causal_and_per_engine():
    values = list(range(1, 41))  # 1..40
    feats = build_features(_frame(values), ["s2"])
    row = feats[feats.cycle == 40].iloc[0]
    assert row.s2_mean10 == np.mean(values[30:40])  # last 10 only
    assert row.s2_mean30 == np.mean(values[10:40])
    assert math.isclose(row.s2_drift, np.mean(values[30:40]) - np.mean(values[:20]))  # vs first 20 cycles
    # Appending future cycles must not change a feature already computed.
    longer = build_features(_frame(values + [1000, 1000]), ["s2"])
    assert longer[longer.cycle == 40].iloc[0].s2_mean10 == row.s2_mean10
    # A second engine starts its own windows.
    both = build_features(pd.concat([_frame(values), _frame([500] * 3, unit=2)]), ["s2"])
    assert both[(both.unit == 2) & (both.cycle == 1)].iloc[0].s2_mean10 == 500


def test_feature_names_match_sql_output():
    feats = build_features(_frame(range(5)), ["s2"])
    assert set(feature_names(["s2"])) <= set(feats.columns)


def test_nasa_score_penalises_late_predictions_more():
    early = metrics.nasa_score([50], [40])  # predicted failure 10 cycles too soon
    late = metrics.nasa_score([50], [60])  # 10 cycles too late
    assert math.isclose(early, math.exp(10 / 13) - 1) and math.isclose(late, math.e - 1) and late > early


def test_alert_metrics_counts():
    result = metrics.alert_metrics(true=[10, 20, 100, 100], predicted=[15, 80, 25, 90])
    assert result["precision"] == 0.5 and result["recall"] == 0.5 and result["engines_near_failure"] == 2


def test_psi_is_near_zero_for_same_distribution_and_high_for_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert metrics.psi(reference, rng.normal(0, 1, 5000)) < 0.02
    assert metrics.psi(reference, rng.normal(1.5, 1, 5000)) > 0.2


def test_drift_report_names_only_the_shifted_column():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)})
    current = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(3, 1, 3000)})
    assert drift_report(reference, current, ["a", "b"])["drifted"] == ["b"]


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _reading(cycle):
    return {"cycle": cycle, **{n: 1.0 for n in (*data.SETTINGS, *data.SENSORS)}}


def _client(monkeypatch, value):
    bundle = {"model": _ConstantModel(value), "sensors": ["s2"], "features": feature_names(["s2"])}
    monkeypatch.setattr(api, "load_bundle", lambda: bundle)
    return TestClient(api.app)


def test_api_returns_rul_and_alert(monkeypatch):
    body = _client(monkeypatch, 12.0).post("/predict", json={"history": [_reading(1), _reading(2)]}).json()
    assert body == {"rul_cycles": 12.0, "maintenance_alert": True, "cycles_seen": 2}
    body = _client(monkeypatch, 500.0).post("/predict", json={"history": [_reading(1)]}).json()
    assert body["rul_cycles"] == data.RUL_CAP and body["maintenance_alert"] is False  # clipped to trained range


def test_api_rejects_empty_or_incomplete_history(monkeypatch):
    client = _client(monkeypatch, 12.0)
    assert client.post("/predict", json={"history": []}).status_code == 422
    assert client.post("/predict", json={"history": [{"cycle": 1, "s2": 1.0}]}).status_code == 422


def test_ready_fails_until_the_model_loads(monkeypatch):
    def missing():
        raise FileNotFoundError("model.joblib")

    monkeypatch.setattr(api, "load_bundle", missing)
    assert TestClient(api.app, raise_server_exceptions=False).get("/ready").status_code == 500
    assert _client(monkeypatch, 12.0).get("/ready").json() == {"status": "ready"}
