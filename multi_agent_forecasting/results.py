from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

METRICS = ["NRMSE", "NMAE", "MASE", "NMBE", "MAE", "RMSE", "MAPE"]


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not pd.notna(value):
        return None
    return value


class ResultStore:
    def __init__(self, directory, configuration):
        self.directory = Path(directory)
        self.records_dir = self.directory / "windows"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        path = self.directory / "run_config.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != configuration:
                raise ValueError("Run configuration changed. Use a new --repeat number or --output directory.")
        else:
            write_json(path, configuration)
        self.records = {}
        for path in sorted(self.records_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (record["score"]["building_id"], record["score"]["window_id"])
            if key in self.records or len(record["predictions"]) != 24:
                raise ValueError(f"Invalid checkpoint: {path.name}")
            self.records[key] = record

    def save(self, window, score, predictions, trajectory):
        record = clean_json({"score": score, "predictions": predictions, "trajectory": trajectory})
        key = window.key
        filename = hashlib.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest()[:24] + ".json"
        write_json(self.records_dir / filename, record)
        self.records[key] = record
        self.export()

    def export(self):
        if not self.records:
            return
        records = [self.records[key] for key in sorted(self.records)]
        scores = pd.DataFrame([record["score"] for record in records])
        predictions = pd.DataFrame([row for record in records for row in record["predictions"]])
        atomic_text(self.directory / "scores.csv", scores.to_csv(index=False))
        atomic_text(self.directory / "predictions.csv", predictions.to_csv(index=False))
        atomic_text(self.directory / "trajectories.jsonl", "".join(
            json.dumps(record["trajectory"], ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        ))
        summary = summarize(scores)
        atomic_text(self.directory / "summary_metrics.csv", summary.to_csv(index=False))
        daytypes = scores.groupby(["dataset", "day_type"])["NRMSE"].agg(["median", "count"])
        atomic_text(self.directory / "summary_daytypes.csv", daytypes.to_csv())


def summarize(scores):
    rows = []
    for dataset, group in scores.groupby("dataset"):
        metrics = METRICS if dataset == "BDG-2" else [key for key in METRICS if key != "MAPE"]
        median = group.groupby("building_id")[metrics].mean().median().to_dict()
        rows.append({
            "dataset": dataset, "n_windows": len(group), "n_buildings": group.building_id.nunique(),
            **median, "failed_share": group.failed.mean(),
            "tokens_in": group.tokens_in.sum(), "tokens_out": group.tokens_out.sum(),
            "seconds_mean": group.seconds.mean(),
            "cost_usd": group.cost_usd.sum() if group.cost_usd.notna().all() else None,
        })
    return pd.DataFrame(rows)
