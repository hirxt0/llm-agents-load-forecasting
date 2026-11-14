import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from agents.scripted import ScriptedClient
from agents.llm import Reply
from agents.supervisor import Ledger, run_supervisor
from agents.workers import run_worker
from data import load_windows
from evaluate import evaluate_runs
from metrics import forecast_metrics
from results import ResultStore, summarize
from tools.environment import ForecastEnv


class PipelineTests(unittest.TestCase):
    def test_metric_scale_and_bias(self):
        history = np.arange(1, 337, dtype=float)
        truth = np.arange(1, 25, dtype=float)
        result = forecast_metrics(truth, truth + 1, history)
        self.assertAlmostEqual(result["NRMSE"], 100 / 168.5)
        self.assertAlmostEqual(result["NMBE"], 100 / 168.5)
        self.assertAlmostEqual(result["MASE"], 1 / 24)
        self.assertEqual(result["MAE"], 1)

    def test_worker_reserves_final_call_for_report(self):
        class GreedyClient:
            def chat(self, system, messages, specs):
                names = {spec["name"] for spec in specs}
                if names == {"submit_report"}:
                    tool, arguments = "submit_report", {"report": "Evidence inspected; final report."}
                else:
                    tool, arguments = "describe_series", {}
                call = {"id": str(len(messages)), "type": "function", "function": {"name": tool, "arguments": json.dumps(arguments)}}
                return Reply({"role": "assistant", "tool_calls": [call]}, [(call["id"], tool, arguments)], ScriptedClient.usage())

        window = load_windows("dev", "all", "LCL", 1)[0][0]
        inputs = window.inputs
        env = ForecastEnv(inputs.building_id, inputs.history, inputs.timestamps, inputs.origin, variant="series")
        ledger = Ledger()
        report = run_worker("series_analyst", "Inspect history", env, GreedyClient(), ledger, 3)
        self.assertIn("final report", report)
        self.assertEqual(ledger.llm_calls, 3)

    def test_failed_api_call_marks_usage_unknown(self):
        class BrokenClient:
            def chat(self, *args):
                raise RuntimeError("API unavailable")

        window = load_windows("dev", "all", "LCL", 1)[0][0]
        inputs = window.inputs
        env = ForecastEnv(inputs.building_id, inputs.history, inputs.timestamps, inputs.origin, variant="series")
        log = run_supervisor(env, BrokenClient())
        self.assertTrue(log["failed"])
        self.assertTrue(log["usage_missing"])
        self.assertEqual(log["llm_calls"], 1)

    def test_same_history_different_truth_cannot_change_forecast(self):
        window = load_windows("dev", "all", "LCL", 1)[0][0]
        alternate = replace(window, truth=np.full(24, 999999.0))
        runs = []
        for candidate in (window, alternate):
            inputs = candidate.inputs
            env = ForecastEnv(inputs.building_id, inputs.history.copy(), inputs.timestamps.copy(), inputs.origin, variant="series")
            log = run_supervisor(env, ScriptedClient("series"))
            self.assertFalse(log["failed"])
            self.assertFalse(hasattr(env, "truth"))
            self.assertEqual(len(env.y), 336)
            runs.append((env.result["pred"], log["events"]))
        np.testing.assert_array_equal(runs[0][0], runs[1][0])
        self.assertEqual(runs[0][1], runs[1][1])
        transcript = json.dumps(runs[0][1])
        self.assertNotIn(window.inputs.building_id, transcript)
        self.assertNotIn("999999", transcript)

    def test_forecast_validation_and_history_bounds(self):
        window = load_windows("dev", "all", "BDG-2", 1)[0][0]
        inputs = window.inputs
        env = ForecastEnv(inputs.building_id, inputs.history, inputs.timestamps, inputs.origin, variant="series")
        for values in ([1] * 23, [-1] * 24, [float("nan")] * 24, [float("inf")] * 24):
            self.assertIn("error", json.loads(env.call("submit_forecast", {"values": values})))
            self.assertFalse(env.done)
        self.assertIn("error", json.loads(env.call("get_values", {"hours_ago_start": 0, "n_hours": 24})))
        self.assertIn("error", json.loads(env.call("weather", {})))
        self.assertEqual(len(json.loads(env.call("get_values", {"hours_ago_start": 1, "n_hours": 72}))), 1)

    def test_aggregation_weights_buildings_equally(self):
        rows = []
        for building, count, error in (("A", 3, 10), ("B", 1, 100)):
            for _ in range(count):
                rows.append({
                    "building_id": building, "dataset": "BDG-2", "failed": False,
                    "NRMSE": error, "NMAE": error, "NMBE": 0, "MASE": 1,
                    "MAE": error, "RMSE": error, "MAPE": error,
                    "tokens_in": 1, "tokens_out": 1, "seconds": 1, "cost_usd": None,
                })
        summary = summarize(pd.DataFrame(rows))
        self.assertEqual(summary.NRMSE.iat[0], 55)
        self.assertIsNone(summary.cost_usd.iat[0])

    def test_resume_after_missing_exports_and_config_mismatch(self):
        window = load_windows("dev", "all", "LCL", 1)[0][0]
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory, {"model": "dry-run"})
            record = {
                "score": {"building_id": window.key[0], "window_id": window.key[1]},
                "predictions": [{}] * 24, "trajectory": {},
            }
            path = Path(directory) / "windows" / "checkpoint.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            recovered = ResultStore(directory, {"model": "dry-run"})
            self.assertIn(window.key, recovered.records)
            with self.assertRaises(ValueError):
                ResultStore(directory, {"model": "other"})

    def test_comparison_recomputes_metrics_and_checks_truth(self):
        windows, _ = load_windows("dev", "all")
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, offset in (("baseline_a", 0.1), ("baseline_b", 0.2)):
                path = Path(directory) / name
                path.mkdir()
                rows = []
                for window in windows:
                    for hour, truth in enumerate(window.truth):
                        rows.append({
                            "building_id": window.key[0], "window_id": window.key[1],
                            "hour": hour, "y_true": truth, "y_pred": truth + offset,
                        })
                pd.DataFrame(rows).to_csv(path / "predictions.csv", index=False)
                paths.append(path)
            output = Path(directory) / "summary"
            evaluate_runs(paths, "dev", "all", output)
            comparisons = pd.read_csv(output / "summary_comparisons.csv")
            pair = comparisons[(comparisons.a == "baseline_a") & (comparisons.b == "baseline_b")]
            self.assertEqual(len(pair), 2)
            np.testing.assert_allclose(pair.ratio, [2, 2])
            self.assertTrue((comparisons.p_holm >= comparisons.p).all())
            corrupted = pd.read_csv(paths[0] / "predictions.csv")
            corrupted.loc[0, "y_true"] += 100
            corrupted.to_csv(paths[0] / "predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "truth does not match"):
                evaluate_runs(paths, "dev", "all", output)


if __name__ == "__main__":
    unittest.main()
