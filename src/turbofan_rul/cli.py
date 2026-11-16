from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="turbofan-rul")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch the NASA C-MAPSS data")
    sub.add_parser("train", help="train, cross-validate and score on the test split")
    sub.add_parser("drift", help="PSI of sensor inputs against the training data")
    args = parser.parse_args()

    from . import data

    if args.command == "download":
        data.download()
        print("ok")
    elif args.command == "train":
        from .train import run

        print(json.dumps(run(), indent=2))
    else:
        from .features import informative_sensors
        from .monitor import drift_report

        train = data.load_train("FD001")
        sensors = informative_sensors(train, data.SENSORS)
        # Reference: half of the training engines. The other half is the no-drift control.
        reference, control = train[train.unit <= 50], train[train.unit > 50]
        report = {
            "FD001 other engines (control)": drift_report(reference, control, sensors),
            "FD001 test (engines stopped before failure)": drift_report(reference, data.load_test("FD001")[0], sensors),
            "FD003 (second fault mode)": drift_report(reference, data.load_train("FD003"), sensors),
        }
        for value in report.values():
            value["psi"] = {k: round(v, 3) for k, v in value["psi"].items()}
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
