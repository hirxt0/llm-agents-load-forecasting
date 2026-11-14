from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONTEXT_RAW, RAW, SHARED


@dataclass(frozen=True)
class ForecastInput:
    building_id: str
    history: np.ndarray
    timestamps: pd.DatetimeIndex
    origin: pd.Timestamp


@dataclass(frozen=True)
class Window:
    inputs: ForecastInput
    truth: np.ndarray
    window_id: int
    dataset: str
    btype: str
    day_type: str

    @property
    def key(self):
        return self.inputs.building_id, self.window_id


def load_windows(split, subset, dataset="both", limit=None):
    path = SHARED / f"windows_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет готовых окон: {path}")
    frame = pd.read_parquet(path)
    required = {"building_id", "dataset", "btype", "day_type", "window_id", "step", "timestamp", "y_clean"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    if subset != "all":
        if split != "test":
            raise ValueError("main/repeat subsets are available only for test")
        column = {"main": "in_main_200", "repeat": "in_repeat_50"}[subset]
        frame = frame[frame[column].eq(True)]
    if dataset != "both":
        frame = frame[frame.dataset.eq(dataset)]
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    windows = []
    for (bid, wid), group in frame.groupby(["building_id", "window_id"], sort=True):
        group = group.sort_values("step")
        if not np.array_equal(group.step.to_numpy(), np.arange(-336, 24)):
            raise ValueError(f"{bid}/{wid}: expected steps -336..23 without duplicates")
        values = group.y_clean.to_numpy(float)
        times = pd.DatetimeIndex(group.timestamp)
        if not np.isfinite(values).all() or (values < 0).any() or values[:336].mean() <= 0:
            raise ValueError(f"{bid}/{wid}: invalid consumption")
        if not times.equals(pd.date_range(times[0], periods=360, freq="h")):
            raise ValueError(f"{bid}/{wid}: non-contiguous hourly timestamps")
        origin = times[336]
        if origin != origin.normalize():
            raise ValueError(f"{bid}/{wid}: forecast must start at midnight")
        history = values[:336].copy()
        history.flags.writeable = False
        windows.append(Window(
            ForecastInput(str(bid), history, times[:336].copy(), origin),
            values[336:].copy(), int(wid), str(group.dataset.iat[0]),
            str(group.btype.iat[0]), str(group.day_type.iat[0]),
        ))
        if limit is not None and len(windows) >= limit:
            break
    if not windows:
        raise ValueError("Нет окон для выбранных параметров")
    return windows, hashlib.sha256(path.read_bytes()).hexdigest()


def check_context(windows, variant):
    missing = set()
    for window in windows:
        if variant in ("calendar", "weather", "full") and window.dataset == "BDG-2":
            path = CONTEXT_RAW / "bdg2_metadata.csv"
            if not path.exists():
                missing.add(str(path))
        if variant in ("weather", "full"):
            bid = window.inputs.building_id
            path = (
                CONTEXT_RAW / "weather_london_openmeteo.json" if window.dataset == "LCL"
                else RAW / "BDG-2" / f"weather_{bid.split('_')[0]}.csv"
            )
            if not path.exists():
                missing.add(str(path))
    if missing:
        raise FileNotFoundError(
            "Для этого варианта нужны исходные файлы контекста одиночного агента:\n"
            + "\n".join(sorted(missing))
            + "\nУкажи CONTEXT_DIR в .env или используй --variant series."
        )
