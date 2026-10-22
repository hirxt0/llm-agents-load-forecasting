"""Прогон агента-прогнозиста (агент сам пишет 24 числа) по окнам BuildingsBench.

Примеры:
  # проверка без API (скриптовая политика вместо LLM)
  python run_forecast_agent.py --dry-run --split dev --limit 4
  # настройка на отладочной выборке
  python run_forecast_agent.py --model gpt-5 --variant weather --split dev
  # тест: лесенка контекста
  python run_forecast_agent.py --model gpt-5 --variant no_tools --split test
  python run_forecast_agent.py --model gpt-5 --variant series   --split test
  python run_forecast_agent.py --model gpt-5 --variant calendar --split test
  python run_forecast_agent.py --model gpt-5 --variant weather  --split test
  python run_forecast_agent.py --model gpt-5 --variant full     --split test
  # проверка на запоминание: те же окна, но даты скрыты
  python run_forecast_agent.py --model gpt-5 --variant weather --split test --masked-dates
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from one_agent_forecasting.src.agent import LLMClient, run_agent
from one_agent_forecasting.src.config import PROCESSED, RESULTS, PRICE_PER_M
from one_agent_forecasting.src.forecast_env import ForecastEnv, VARIANT_TOOLS
from one_agent_forecasting.src.metrics import forecast_metrics

LOCK = threading.Lock()


class ScriptedLLM:
    """Заглушка вместо LLM для проверки кода: смотрит профили и сдаёт среднее за 2 недели."""
    provider = "dry-run"

    def __init__(self):
        self.step = 0

    def chat(self, system, messages, specs):
        names = [s["name"] for s in specs]
        self.step += 1
        if self.step == 1 and "reference_profiles" in names:
            name, args = "reference_profiles", {}
        elif "calendar" in names and self.step == 2:
            name, args = "calendar", {}
        else:
            prof = None
            for m in reversed(messages):
                if m.get("role") == "tool" and "mean_same_weekday_last_2_weeks" in m["content"]:
                    prof = json.loads(m["content"])["mean_same_weekday_last_2_weeks"]
                    break
            name, args = "submit_forecast", {"values": prof or [0.0] * 24, "reasoning": "scripted"}
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"id": str(self.step), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        return "", [(str(self.step), name, args)], msg, {"in": 0, "out": 0, "reasoning": 0, "cached": 0}

    def tool_results(self, results):
        return [{"role": "tool", "tool_call_id": i, "content": c} for i, c in results]


def cost_usd(model, log):
    p = PRICE_PER_M.get(model)
    if not p:
        return np.nan
    fresh = log["tokens_in"] - log["tokens_cached"]
    return (fresh * p["input"] + log["tokens_cached"] * p["cached_input"] + log["tokens_out"] * p["output"]) / 1e6


def run_one(a, key, g, out_dir):
    bid, wid = key
    g = g.sort_values("step")
    hist, fut = g[g.step < 0], g[g.step >= 0]
    y_hist, y_true = hist.y_clean.to_numpy(), fut.y_clean.to_numpy()
    env = ForecastEnv(bid, y_hist, hist.timestamp.to_numpy(), fut.timestamp.iat[0],
                      variant=a.variant, weather_mode=a.weather_mode, masked_dates=a.masked_dates)
    llm = ScriptedLLM() if a.dry_run else LLMClient(a.provider, a.model, reasoning_effort=a.reasoning_effort)
    log = run_agent(env, llm, max_steps=a.max_steps)
    if env.done:
        pred = env.result["pred"]
    else:                                      # сбой: прогноз «как вчера», сбой считаем отдельно
        pred = y_hist[-24:]
    row = {"building_id": bid, "window_id": wid, "dataset": g.dataset.iat[0], "btype": g.btype.iat[0], "day_type": g.day_type.iat[0],
           "variant": a.variant, "masked_dates": a.masked_dates, "weather_mode": a.weather_mode,
           "model": a.model if not a.dry_run else "dry-run", "reasoning_effort": a.reasoning_effort,
           **forecast_metrics(y_true, pred, y_hist),
           **{k: v for k, v in log.items() if k != "messages"},
           "invalid_submissions": env.n_invalid,
           "cost_usd": cost_usd(a.model, log),
           "tools_used": ",".join(x["tool"] for x in env.actions),
           "reasoning": env.result["reasoning"] if env.done else ""}
    with LOCK:
        for name, df in [("scores.csv", pd.DataFrame([row])),
                         ("predictions.csv", pd.DataFrame({"building_id": bid, "window_id": wid, "hour": range(24),
                                                           "y_true": y_true, "y_pred": pred}))]:
            p = out_dir / name
            df.to_csv(p, mode="a", header=not p.exists(), index=False)
        with open(out_dir / "trajectories.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"building_id": bid, "window_id": wid, "actions": env.actions,
                                "messages": log["messages"], **{k: v for k, v in row.items()}},
                               ensure_ascii=False, default=str) + "\n")
    print(f"{bid:32s} w{wid} NRMSE={row['NRMSE']:.1f}% MASE={row['MASE']:.2f} calls={log['llm_calls']} "
          f"tok={log['tokens_in']}+{log['tokens_out']} ${row['cost_usd']:.4f} {log['seconds']}s"
          + (" FAILED" if log["failed"] else "") + (f" {log['error']}" if log["error"] else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--reasoning-effort", default="low", choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--variant", default="weather", choices=list(VARIANT_TOOLS))
    ap.add_argument("--weather-mode", default="noisy", choices=["none", "noisy", "perfect"])
    ap.add_argument("--masked-dates", action="store_true")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--subset", default="all", choices=["all", "main", "repeat"],
                    help="main — основные 200 окон, repeat — 50 окон для повтора (только test)")
    ap.add_argument("--limit", type=int, default=None, help="взять только первые N окон (для быстрых проверок)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=0, help="номер повторного запуска (отдельная папка)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    win = pd.read_parquet(PROCESSED / ("windows_dev.parquet" if a.split == "dev" else "windows.parquet"))
    if a.subset != "all":
        win = win[win[{"main": "in_main_200", "repeat": "in_repeat_50"}[a.subset]]]
    groups = list(win.groupby(["building_id", "window_id"]))
    if a.limit:
        groups = groups[: a.limit]

    tag = "dry-run" if a.dry_run else f"{a.model}-{a.reasoning_effort}"
    name = f"fc_{tag}_{a.variant}_{a.weather_mode}{'_masked' if a.masked_dates else ''}{f'_r{a.repeat}' if a.repeat else ''}_{a.split}"
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(a), indent=2))
    done = set()
    if (out_dir / "scores.csv").exists():      # продолжение прерванного прогона
        s = pd.read_csv(out_dir / "scores.csv")
        done = set(zip(s.building_id, s.window_id))
    todo = [(k, g) for k, g in groups if k not in done]
    print(f"{name}: окон {len(groups)}, уже готово {len(done)}, осталось {len(todo)}")

    with ThreadPoolExecutor(max_workers=1 if a.dry_run else a.workers) as ex:
        list(ex.map(lambda kg: run_one(a, kg[0], kg[1], out_dir), todo))

    s = pd.read_csv(out_dir / "scores.csv")
    print(s.groupby("dataset")[["NRMSE", "NMAE", "MASE", "NMBE"]].median().round(3))
    print("итого: окон", len(s), "| сбоев", round(s.failed.mean(), 3), "| стоимость, $", round(s.cost_usd.sum(), 3))


if __name__ == "__main__":
    main()
