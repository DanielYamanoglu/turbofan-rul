"""Render the README figure from the trained model: predicted vs true RUL, and three engines over time."""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from turbofan_rul import data
from turbofan_rul.metrics import ALERT_HORIZON
from turbofan_rul.train import MODEL_DIR, last_cycle, predict

OUT = Path(__file__).resolve().parents[1] / "assets" / "predictions.png"


def main() -> None:
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    test, truth = data.load_test()
    predictions = predict(bundle, test)
    final = last_cycle(predictions).loc[truth.index]
    capped = truth.clip(upper=data.RUL_CAP)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    left.scatter(capped, final, s=22, color="#0284c7", alpha=0.8)
    left.plot([0, data.RUL_CAP], [0, data.RUL_CAP], color="#475569", lw=1, ls="--", label="perfect prediction")
    left.axvline(ALERT_HORIZON, color="#d97706", lw=1, ls=":")
    left.axhline(ALERT_HORIZON, color="#d97706", lw=1, ls=":", label=f"alert horizon ({ALERT_HORIZON} cycles)")
    left.set(title="100 unseen engines, at their last observed cycle", xlabel="true RUL [cycles]", ylabel="predicted RUL [cycles]")
    left.legend(frameon=False)

    for unit, color in zip((20, 34, 81), ("#0284c7", "#16a34a", "#d97706")):
        engine = predictions[predictions.unit == unit]
        true_now = truth[unit] + (engine.cycle.max() - engine.cycle)
        right.plot(engine.cycle, engine.rul, color=color, label=f"engine {unit}: predicted")
        right.plot(engine.cycle, true_now.clip(upper=data.RUL_CAP), color=color, ls="--", lw=1)
    right.set(title="Prediction over an engine's life (dashed: true)", xlabel="cycle", ylabel="RUL [cycles]")
    right.legend(frameon=False)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
