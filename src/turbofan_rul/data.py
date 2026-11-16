"""Download and load the NASA C-MAPSS turbofan degradation data."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
# NASA Prognostics Center of Excellence data repository.
URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

SETTINGS = [f"op{i}" for i in range(1, 4)]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]
# Engines far from failure look alike, so the target is capped (the usual piecewise-linear RUL).
RUL_CAP = 125


def download(raw_dir: Path = RAW_DIR) -> None:
    """Fetch the archive and extract only the plain-text data files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner_name = next(name for name in outer.namelist() if name.endswith(".zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for name in inner.namelist():
        base = Path(name).name
        if base.endswith(".txt") and base.split("_")[0] in {"train", "test", "RUL"}:
            (raw_dir / base).write_bytes(inner.read(name))


def read_trajectories(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(frame: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Training engines run to failure: RUL is the distance to each engine's last cycle."""
    last = frame.groupby("unit")["cycle"].transform("max")
    return frame.assign(rul=(last - frame["cycle"]).clip(upper=cap))


def load_train(subset: str = "FD001", raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return add_rul(read_trajectories(raw_dir / f"train_{subset}.txt"))


def load_test(subset: str = "FD001", raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Test engines stop before failure; the true RUL at the last cycle comes in a separate file."""
    frame = read_trajectories(raw_dir / f"test_{subset}.txt")
    truth = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None)[0]
    truth.index = sorted(frame["unit"].unique())
    return frame, truth
