"""Проба на запоминание: помнит ли LLM данные BuildingsBench из предобучения.

Модель получает ОДИНАКОВУЮ скудную информацию (тип здания, площадь, город, дата) и просит
24 значения. Истории потребления она не видит. Две версии запроса:
  identity — с настоящим названием датасета и здания (BDG-2 «Fox_office_Israel», London LCL «MAC000123»);
  anonymous — то же самое, но здание и датасет обезличены.
Если запомненного нет, качество обеих версий одинаково (и плохое). Заметный выигрыш identity
над anonymous — признак утечки через предобучение.

  python probe_memorization.py --model gpt-5 --limit 60
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import one_agent_forecasting.src.context as context
from one_agent_forecasting.src.agent import LLMClient
from one_agent_forecasting.src.config import PROCESSED, RESULTS
from one_agent_forecasting.src.metrics import forecast_metrics

DATASET_NAME = {"BDG-2": "Building Data Genome Project 2 (BDG-2)",
                "LCL": "Low Carbon London smart meter dataset (UK Power Networks)"}
SUBMIT = [{"name": "submit_forecast", "description": "Submit 24 hourly kWh values.",
           "parameters": {"type": "object", "properties": {
               "values": {"type": "array", "items": {"type": "number"}, "minItems": 24, "maxItems": 24}},
               "required": ["values"]}}]


def prompt(bid, dataset, day, identity):
    info = context.building_info(bid, reveal_city=True)
    if identity:
        who = (f"building '{bid.split('@')[0]}' from the public dataset {DATASET_NAME[dataset]}")
    else:
        who = f"an anonymous building ({info['building_id']})"
    desc = ", ".join(f"{k}: {v}" for k, v in info.items() if k not in ("building_id",))
    return (f"Give your best estimate of the hourly electricity consumption (kWh) of {who} "
            f"on {day:%Y-%m-%d (%A)}, hours 00-23. Known facts: {desc}. "
            f"If you have seen this building's data before, use that knowledge. Call submit_forecast.")


def ask(llm, text):
    msgs = [{"role": "user", "content": text}]
    for _ in range(3):
        _, calls, msg, _ = llm.chat("You are an energy analyst.", msgs, SUBMIT)
        if calls:
            v = np.asarray(calls[0][2].get("values", []), float)
            if v.shape == (24,) and np.isfinite(v).all():
                return np.maximum(v, 0)
        msgs += [msg, {"role": "user", "content": "Call submit_forecast with exactly 24 numbers."}]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    llm = LLMClient("openai", a.model, reasoning_effort=a.reasoning_effort)
    win = pd.read_parquet(PROCESSED / "windows.parquet")
    groups = list(win.groupby(["building_id", "window_id"]))
    groups = groups[:: max(1, len(groups) // a.limit)][: a.limit]
    rows = []
    for (bid, wid), g in groups:
        g = g.sort_values("step")
        hist, fut = g[g.step < 0], g[g.step >= 0]
        day = fut.timestamp.iat[0]
        for identity in (True, False):
            pred = ask(llm, prompt(bid, g.dataset.iat[0], day, identity))
            if pred is None:
                continue
            m = forecast_metrics(fut.y_clean.to_numpy(), pred, hist.y_clean.to_numpy())
            rows.append({"building_id": bid, "window_id": wid, "dataset": g.dataset.iat[0],
                         "identity": identity, **m})
            print(bid, wid, "identity" if identity else "anonymous", f"NRMSE={m['NRMSE']:.1f}%", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS / f"memorization_probe_{a.model}.csv"
    df.to_csv(out, index=False)
    p = df.pivot_table(index=["building_id", "window_id"], columns="identity", values="NRMSE").dropna()
    print("\nNRMSE, %: identity vs anonymous (меньше — лучше)")
    print(df.groupby(["dataset", "identity"]).NRMSE.median().unstack().round(3))
    if len(p) >= 10:
        stat = wilcoxon(p[True], p[False])
        print(f"медиана разницы identity - anonymous: {(p[True] - p[False]).median():.3f}, "
              f"Wilcoxon p = {stat.pvalue:.4f}  (p < 0.05 и разница < 0 — признак запоминания)")


if __name__ == "__main__":
    main()
