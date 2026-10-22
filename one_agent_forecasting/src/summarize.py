"""Сводка всех систем (агенты, мультиагент, бейзлайны) по общим метрикам и статистике.

Источники: results/<папка>_<split>/predictions.csv с колонками
building_id, window_id, hour, y_true, y_pred (любая система команды: fc_* — агенты,
bl_* — бейзлайны, ma_* — мультиагентная система). Метрики пересчитываются из прогнозов.
Эталоны snaive_24 («как вчера») и snaive_168 («как неделю назад») считаются здесь же.

Выход (results/summary_<split>_*.csv) и печать:
  1) метрики по датасетам: медиана по зданиям (сначала среднее по окнам здания);
  2) NRMSE по типам дней (будний, выходной, праздник, погодная аномалия);
  3) сравнения: лесенка агента, агенты против эталонов и бейзлайнов (отношение NRMSE,
     95% ДИ кластерным бутстрепом по зданиям, Wilcoxon по зданиям, поправка Холма);
  4) проверка на утечку: настоящие даты против скрытых (тест эквивалентности ±5%);
  5) разброс между повторными запусками; 6) стоимость и надёжность.
"""
import argparse

import numpy as np
import pandas as pd

from one_agent_forecasting.src.config import PROCESSED, RESULTS
from one_agent_forecasting.src.metrics import forecast_metrics
from one_agent_forecasting.src.stats import paired_compare, holm, equivalent

LADDER = ["no_tools", "series", "calendar", "weather", "full"]
METRICS = ["NRMSE", "NMAE", "MASE", "NMBE", "MAE", "RMSE", "MAPE"]


def load_windows(split):
    w = pd.read_parquet(PROCESSED / ("windows.parquet" if split == "test" else "windows_dev.parquet"))
    hist = {k: g.sort_values("step").y_clean.to_numpy() for k, g in w[w.step < 0].groupby(["building_id", "window_id"])}
    meta = w[w.step == 0].set_index(["building_id", "window_id"])[["dataset", "day_type"]]
    return w, hist, meta


def reference_predictions(w):
    """Эталоны «как вчера» и «как неделю назад» для всех окон."""
    out = []
    for (b, i), g in w.groupby(["building_id", "window_id"]):
        g = g.sort_values("step")
        y, truth = g.y_clean.to_numpy(), g[g.step >= 0].y_clean.to_numpy()
        for name, pred in [("snaive_24", y[312:336]), ("snaive_168", y[168:192])]:
            out.append(pd.DataFrame({"method": name, "building_id": b, "window_id": i, "hour": range(24),
                                     "y_true": truth, "y_pred": pred}))
    return pd.concat(out, ignore_index=True)


def load_systems(split):
    preds, scores = [], []
    for d in sorted(RESULTS.glob(f"*_{split}")):
        if not (d / "predictions.csv").exists() or "dry-run" in d.name:
            continue
        m = d.name.removesuffix(f"_{split}").removeprefix("fc_")
        preds.append(pd.read_csv(d / "predictions.csv").assign(method=m))
        if (d / "scores.csv").exists():
            scores.append(pd.read_csv(d / "scores.csv").assign(method=m))
    return (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
            pd.concat(scores, ignore_index=True) if scores else pd.DataFrame())


def per_window(pred, hist, meta):
    rows = []
    for (m, b, i), g in pred.groupby(["method", "building_id", "window_id"]):
        if (b, i) not in hist:
            continue                                   # окно не из текущей выборки
        g = g.sort_values("hour")
        rows.append({"method": m, "building_id": b, "window_id": i, **meta.loc[(b, i)].to_dict(),
                     **forecast_metrics(g.y_true.to_numpy(), g.y_pred.to_numpy(), hist[(b, i)])})
    return pd.DataFrame(rows)


def is_repeat(m):
    last = m.split("_")[-1]
    return last.startswith("r") and last[1:].isdigit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    w, hist, meta = load_windows(a.split)
    systems, scores = load_systems(a.split)
    if systems.empty:
        print("нет прогонов для", a.split)
        return
    keys = set(zip(systems.building_id, systems.window_id))
    ref = reference_predictions(w)
    ref = ref[[k in keys for k in zip(ref.building_id, ref.window_id)]]
    pw = per_window(pd.concat([systems, ref], ignore_index=True), hist, meta)
    pw.to_csv(RESULTS / f"summary_{a.split}_windows.csv", index=False)

    # 1) метрики по датасетам: сначала среднее по окнам здания, потом медиана по зданиям
    tables = []
    for ds, g in pw.groupby("dataset"):
        cols = METRICS if ds == "BDG-2" else [c for c in METRICS if c != "MAPE"]
        t = g.groupby(["method", "building_id"])[cols].mean().groupby("method").median().round(3)
        t.insert(0, "n_windows", g.groupby("method").size())
        t = t.sort_values("NRMSE")
        tables.append(t.assign(dataset=ds))
        name = "коммерческие (BDG-2)" if ds == "BDG-2" else "жилые (LCL)"
        print(f"\n=== {name}: медиана по зданиям, меньше — лучше (NRMSE — основная)\n{t.to_string()}")
    pd.concat(tables).to_csv(RESULTS / f"summary_{a.split}_metrics.csv")

    # 2) по типам дней
    dt = pw.groupby(["dataset", "day_type", "method"]).NRMSE.median().unstack("day_type").round(2)
    print(f"\n=== NRMSE по типам дней (медиана по окнам)\n{dt.to_string()}")
    dt.to_csv(RESULTS / f"summary_{a.split}_daytypes.csv")

    # 3) сравнения
    methods = sorted(pw.method.unique())
    agents = [m for m in methods if not m.startswith(("snaive", "bl_"))]
    main_runs = [m for m in agents if "masked" not in m and not is_repeat(m)]
    pairs = []
    for prefix in sorted({m.split("_")[0] for m in main_runs}):
        steps = [m for v in LADDER for m in main_runs if m.startswith(f"{prefix}_{v}_")]
        pairs += list(zip(steps, steps[1:]))
    refs = ["snaive_168"] + [m for m in methods if m.startswith("bl_")]
    pairs += [(r, m) for m in main_runs for r in refs]
    res = []
    for x, y in pairs:
        for ds in ("BDG-2", "LCL"):
            r = paired_compare(pw[pw.dataset == ds], x, y)
            if r:
                res.append({**r, "dataset": ds})
    if res:
        cmp = pd.DataFrame(res)
        cmp["p_holm"] = holm(cmp.p)
        cmp["ci95"] = cmp.ci95.map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
        cmp = cmp.drop(columns="ci90")
        print("\n=== Сравнения: отношение NRMSE b/a по зданиям (< 1 — b лучше), 95% ДИ, Wilcoxon + Холм")
        print(cmp[["dataset", "a", "b", "n_buildings", "ratio", "ci95", "p", "p_holm"]].round(4).to_string(index=False))
        cmp.to_csv(RESULTS / f"summary_{a.split}_comparisons.csv", index=False)

    # 4) утечка: настоящие даты против скрытых
    for m in agents:
        real = m.removesuffix("_masked")
        if m.endswith("_masked") and real in methods:
            for ds in ("BDG-2", "LCL"):
                r = paired_compare(pw[pw.dataset == ds], real, m)
                if r:
                    eq = equivalent(r["ci90"])
                    print(f"\n=== Утечка [{ds}] {real} → скрытые даты: отношение {r['ratio']:.3f}, "
                          f"95% ДИ [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}], p={r['p']:.4f}; "
                          f"эквивалентны в пределах ±5%: {'да' if eq else 'нет'}. "
                          "Заметно хуже со скрытыми датами — агенту помогает память о датах (нижняя оценка утечки).")

    # 5) разброс между повторами
    for m in agents:
        base = m.rsplit("_r", 1)[0]
        if is_repeat(m) and base in methods:
            p = pw[pw.method.isin([base, m])].pivot_table(index=["building_id", "window_id"], columns="method",
                                                          values="NRMSE").dropna()
            lr = np.log(p[m] / p[base])
            print(f"\n=== Повтор {m}: окон {len(p)}, SD лог-отношения между запусками {lr.std():.3f}, "
                  f"медиана |изменения| {np.expm1(lr.abs().median()):.1%}")

    # 6) стоимость и надёжность
    if not scores.empty:
        c = scores.groupby("method").agg(n=("failed", "size"), failed=("failed", "mean"),
                                         llm_calls=("llm_calls", "mean"), tokens_in=("tokens_in", "mean"),
                                         tokens_out=("tokens_out", "mean"), seconds=("seconds", "mean"),
                                         usd_per_forecast=("cost_usd", "mean"), usd_total=("cost_usd", "sum")).round(4)
        print(f"\n=== Стоимость и надёжность\n{c.to_string()}")
        c.to_csv(RESULTS / f"summary_{a.split}_cost.csv")


if __name__ == "__main__":
    main()
