from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, APIStatusError

from agents.llm import LLMClient
from agents.scripted import ScriptedClient
from agents.supervisor import run_supervisor
from config import RESULTS, ROOT, Settings
from data import check_context, load_windows
from metrics import forecast_metrics
from results import ResultStore, summarize
from tools.environment import ForecastEnv


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    command = argparse.ArgumentParser(description="Мультиагентный прогноз на 24 часа: BDG-2 и LCL, GPT-5 через apinet.cloud.")
    command.add_argument("--check-api", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--split", choices=["dev", "test"], default="dev")
    command.add_argument("--subset", choices=["all", "main", "repeat"])
    command.add_argument("--dataset", choices=["both", "BDG-2", "LCL"], default="both")
    command.add_argument("--variant", choices=["series", "calendar", "weather", "full"], default="series")
    command.add_argument("--masked-dates", action="store_true")
    command.add_argument("--limit", type=positive)
    command.add_argument("--max-steps", type=positive, default=8)
    command.add_argument("--worker-steps", type=positive, default=5)
    command.add_argument("--repeat", type=int, default=0)
    command.add_argument("--retry-failed", action="store_true")
    command.add_argument("--model")
    command.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    command.add_argument("--output", type=Path)
    return command


def source_hash():
    digest = hashlib.sha256()
    files = [*ROOT.glob("*.py"), *(ROOT / "agents").glob("*.py"), *(ROOT / "tools").glob("*.py")]
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_window(window, args, settings, client, store):
    inputs = window.inputs
    env = ForecastEnv(
        inputs.building_id, inputs.history.copy(), inputs.timestamps.copy(), inputs.origin,
        variant=args.variant, weather_mode="noisy", masked_dates=args.masked_dates,
    )
    log = run_supervisor(env, client, args.max_steps, args.worker_steps)
    prediction = env.result["pred"] if env.done else inputs.history[-24:]
    metrics = forecast_metrics(window.truth, prediction, inputs.history)
    if window.dataset == "LCL":
        metrics["MAPE"] = None
    error = log["error"]
    if error and settings.api_key:
        error = error.replace(settings.api_key, "[REDACTED]")
    score = {
        "building_id": inputs.building_id, "window_id": window.window_id,
        "dataset": window.dataset, "btype": window.btype, "day_type": window.day_type,
        "variant": args.variant, "weather_mode": "noisy", "masked_dates": args.masked_dates,
        "model": "dry-run" if args.dry_run else settings.model,
        "reasoning_effort": settings.reasoning_effort, **metrics,
        **{key: value for key, value in log.items() if key not in ("events", "reports", "error")},
        "error": error, "cost_usd": 0.0 if args.dry_run else settings.cost_usd(log),
        "invalid_submissions": env.n_invalid,
        "tools_used": ",".join(action["tool"] for action in env.actions),
        "reasoning": env.result["reasoning"] if env.done else "Fallback: yesterday",
    }
    predictions = [
        {"building_id": inputs.building_id, "window_id": window.window_id,
         "hour": hour, "y_true": float(truth), "y_pred": float(pred)}
        for hour, (truth, pred) in enumerate(zip(window.truth, prediction))
    ]
    store.save(window, score, predictions, {
        "building_id": inputs.building_id, "window_id": window.window_id,
        "events": log["events"], "reports": log["reports"], "actions": env.actions,
        "failed": log["failed"], "error": error,
    })
    status = f"FAILED: {error}" if log["failed"] else "OK"
    print(f"{inputs.building_id} w{window.window_id}: NRMSE={metrics['NRMSE']:.2f}% "
          f"calls={log['llm_calls']} tokens={log['tokens_in']}+{log['tokens_out']} {status}", flush=True)
    return not log["failed"]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    command = parser()
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        command.print_help()
        return 0
    args = command.parse_args(arguments)
    if args.repeat < 0:
        command.error("--repeat must be non-negative")
    if args.masked_dates and args.variant == "full":
        command.error("--masked-dates is incompatible with --variant full")
    args.subset = args.subset or ("main" if args.split == "test" else "all")
    settings = Settings()
    if args.model:
        settings = replace(settings, model=args.model)
    if args.reasoning_effort:
        settings = replace(settings, reasoning_effort=args.reasoning_effort)
    if args.check_api:
        print(json.dumps(LLMClient(settings).check(), ensure_ascii=False, indent=2))
        return 0
    windows, windows_hash = load_windows(args.split, args.subset, args.dataset, args.limit)
    check_context(windows, args.variant)
    tag = "dry-run" if args.dry_run else f"{settings.model}-{settings.reasoning_effort}"
    tag = re.sub(r"[^A-Za-z0-9_.-]", "-", tag)
    name = (f"ma_{tag}_{args.variant}_noisy_{args.dataset}_{args.subset}"
            f"{'_masked' if args.masked_dates else ''}{f'_r{args.repeat}' if args.repeat else ''}_{args.split}")
    configuration = {
        "model": "dry-run" if args.dry_run else settings.model,
        "base_url": settings.base_url, "reasoning_effort": settings.reasoning_effort,
        "max_completion_tokens": settings.max_completion_tokens,
        "split": args.split, "subset": args.subset, "dataset": args.dataset,
        "variant": args.variant, "weather_mode": "noisy", "masked_dates": args.masked_dates,
        "max_steps": args.max_steps, "worker_steps": args.worker_steps,
        "repeat": args.repeat, "dry_run": args.dry_run,
        "windows_sha256": windows_hash, "source_sha256": source_hash(),
    }
    directory = args.output or RESULTS / name
    if not directory.is_absolute():
        directory = ROOT / directory
    store = ResultStore(directory, configuration)
    client = ScriptedClient(args.variant) if args.dry_run else LLMClient(settings)
    remaining = [
        window for window in windows
        if window.key not in store.records
        or (args.retry_failed and store.records[window.key]["score"]["failed"])
    ]
    print(f"Окон: {len(windows)}; готово: {len(windows) - len(remaining)}; осталось: {len(remaining)}")
    print(f"Результаты: {directory}", flush=True)
    any_failed = False
    try:
        for window in remaining:
            success = run_window(window, args, settings, client, store)
            any_failed |= not success
            if not success:
                print("Прогон остановлен после сбоя. Сохранён fallback «как вчера»; подробности в scores.csv.")
                break
    finally:
        if hasattr(client, "client"):
            client.client.close()
    store.export()
    if store.records:
        scores = pd.DataFrame([record["score"] for record in store.records.values()])
        print(summarize(scores).round(3).to_string(index=False))
        any_failed |= bool(scores.failed.any())
    return 1 if any_failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, APIConnectionError, APIStatusError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nПрервано. Готовые окна сохранены; повтори команду для продолжения.")
        raise SystemExit(130)
