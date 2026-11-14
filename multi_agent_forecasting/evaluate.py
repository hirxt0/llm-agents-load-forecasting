from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS
from data import load_windows
from metrics import forecast_metrics
from results import METRICS, atomic_text
from stats import holm, paired_compare


def evaluate_runs(paths, split, subset, output):
    windows, _ = load_windows(split, subset)
    lookup = {window.key: window for window in windows}
    rows = []
    methods = set()
    for path in paths:
        directory = Path(path).resolve()
        config_path = directory / "run_config.json"
        configuration = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        if "dry-run" in directory.name or configuration.get("dry_run"):
            raise ValueError("Offline checks must not be mixed with real experiments")
        method = directory.name
        if method in methods:
            raise ValueError(f"Duplicate method: {method}")
        methods.add(method)
        predictions = pd.read_csv(directory / "predictions.csv")
        for (bid, wid), group in predictions.groupby(["building_id", "window_id"]):
            window = lookup.get((bid, int(wid)))
            if window is None:
                continue
            group = group.sort_values("hour")
            if not np.array_equal(group.hour.to_numpy(), np.arange(24)):
                raise ValueError(f"{method}: expected exactly hours 0..23 for {bid}/{wid}")
            prediction = group.y_pred.to_numpy(float)
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise ValueError(f"{method}: invalid forecast for {bid}/{wid}")
            if not np.allclose(group.y_true.to_numpy(float), window.truth, atol=1e-7, rtol=1e-7):
                raise ValueError(f"{method}: truth does not match shared windows for {bid}/{wid}")
            row = {
                "method": method, "building_id": bid, "window_id": int(wid),
                "dataset": window.dataset, "day_type": window.day_type,
                **forecast_metrics(window.truth, prediction, window.inputs.history),
            }
            if window.dataset == "LCL":
                row["MAPE"] = None
            rows.append(row)
    if not rows:
        raise ValueError("No matching predictions")
    frame = pd.DataFrame(rows)
    references = []
    for bid, wid in frame[["building_id", "window_id"]].drop_duplicates().itertuples(index=False, name=None):
        window = lookup[(bid, wid)]
        for name, prediction in (
            ("snaive_24", window.inputs.history[-24:]),
            ("snaive_168", window.inputs.history[-168:-144]),
        ):
            row = {
                "method": name, "building_id": bid, "window_id": wid,
                "dataset": window.dataset, "day_type": window.day_type,
                **forecast_metrics(window.truth, prediction, window.inputs.history),
            }
            if window.dataset == "LCL":
                row["MAPE"] = None
            references.append(row)
    frame = pd.concat([frame, pd.DataFrame(references)], ignore_index=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "summary_windows.csv", frame.to_csv(index=False))
    summaries = []
    comparisons = []
    for dataset, group in frame.groupby("dataset"):
        columns = METRICS if dataset == "BDG-2" else [column for column in METRICS if column != "MAPE"]
        table = group.groupby(["method", "building_id"])[columns].mean().groupby("method").median()
        table["n_windows"] = group.groupby("method").size()
        table["n_buildings"] = group.groupby("method").building_id.nunique()
        table["dataset"] = dataset
        summaries.append(table.reset_index())
        for left, right in combinations(sorted(group.method.unique()), 2):
            comparison = paired_compare(group, left, right)
            if comparison:
                comparison["dataset"] = dataset
                comparisons.append(comparison)
    summary = pd.concat(summaries, ignore_index=True)
    atomic_text(output / "summary_metrics.csv", summary.to_csv(index=False))
    daytypes = frame.groupby(["dataset", "day_type", "method"]).NRMSE.agg(["median", "count"])
    atomic_text(output / "summary_daytypes.csv", daytypes.to_csv())
    if comparisons:
        comparisons = pd.DataFrame(comparisons)
        comparisons["p_holm"] = holm(comparisons.p)
        atomic_text(output / "summary_comparisons.csv", comparisons.to_csv(index=False))
    print(summary.round(3).to_string(index=False))
    print(f"Summary: {output}")


def main():
    parser = argparse.ArgumentParser(description="Общая оценка ma_*, fc_*, bl_* на одинаковых окнах.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--subset", choices=["all", "main", "repeat"])
    parser.add_argument("--output", type=Path, default=RESULTS / "comparison")
    args = parser.parse_args()
    evaluate_runs(args.runs, args.split, args.subset or ("main" if args.split == "test" else "all"), args.output)


if __name__ == "__main__":
    main()
