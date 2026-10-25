"""Классические бейзлайны на общих окнах команды (shared/windows_*.parquet).

Запуск из корня репозитория:
    python -m baselines.run_baselines --split test
    python -m baselines.run_baselines --split dev --models snaive_24 lgbm_weather

Модели (имя папки результата: results/bl_<модель>_<split>/):
  snaive_24        как вчера: последние 24 ч истории
  snaive_168       как неделю назад: те же часы 7 суток назад
  persistence_ens  Persistence Ensemble из BuildingsBench (AveragePersistence):
                   среднее того же часа за 7 предыдущих суток
  ets              AutoETS, сезон 24 (statsforecast)
  mstl             MSTL, сезоны 24 и 168, тренд AutoETS (statsforecast)
  lgbm             LightGBM локально по окну: лаги 24/48/72/168, среднее за предыдущие сутки,
                   час, день недели, флаг праздника; прямой прогноз на 24 ч (все лаги >= 24)
  lgbm_weather     то же + температура: история — факт, день прогноза — «прогноз с ошибкой»
                   (та же функция и то же зерно, что у агентов: context.weather(mode="noisy"))
  automl           выбор лучшей из snaive_24/snaive_168/persistence_ens/ets/mstl/lgbm по MAE
                   на последних 2 сутках истории (без доступа к прогнозируемым суткам)

Правила: каждая модель видит только 336 ч истории своего окна; гиперпараметры фиксированы
(на тесте ничего не подбирается); отрицательные прогнозы обрезаются до 0; если модель упала —
подставляется snaive_24, и это записывается в run_info.json.
"""
import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from one_agent_forecasting.src import context
from one_agent_forecasting.src.config import WINDOWS, RESULTS, HISTORY, HORIZON

warnings.filterwarnings("ignore")
MODELS = ["snaive_24", "snaive_168", "persistence_ens", "ets", "mstl", "lgbm", "lgbm_weather", "automl"]
AUTOML_CANDIDATES = ["snaive_24", "snaive_168", "persistence_ens", "ets", "mstl", "lgbm"]
LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                   verbose=-1, random_state=0)


# ---------------------------------------------------------------- признаки для LightGBM

def _features(y, idx, bid, temp=None):
    """y — ряд длиной len(idx) (будущие значения = NaN); все лаги >= 24, поэтому будущее не нужно."""
    s = pd.Series(y, index=idx)
    df = pd.DataFrame({"y": s})
    for lag in (24, 48, 72, 168):
        df[f"lag{lag}"] = s.shift(lag)
    df["mean_prev_day"] = s.shift(24).rolling(24).mean()
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    days = {d: bool(context.holiday_name(bid, d)) for d in idx.normalize().unique()}
    df["holiday"] = [days[d] for d in idx.normalize()]
    if temp is not None:
        df["temp"] = temp
    return df


def _lgbm(y, ts, bid, origin, weather):
    import lightgbm as lgb
    idx = pd.DatetimeIndex(ts).append(pd.date_range(origin, periods=HORIZON, freq="h"))
    temp = None
    if weather:
        hist_t, fut_t = context.weather(bid, ts, idx[-HORIZON:], mode="noisy", key=(bid, str(origin)))
        temp = np.r_[hist_t.interpolate(limit_direction="both").to_numpy(), fut_t]
    df = _features(np.r_[y, np.full(HORIZON, np.nan)], idx, bid, temp)
    X = [c for c in df.columns if c != "y"]
    train = df.iloc[:len(y)].dropna()
    model = lgb.LGBMRegressor(**LGBM_PARAMS).fit(train[X], train["y"])
    return model.predict(df.iloc[len(y):][X])


# ---------------------------------------------------------------- модели

def predict(model, y, ts, bid, origin):
    if model == "snaive_24":
        return y[-24:]
    if model == "snaive_168":
        return y[-168:-144]
    if model == "persistence_ens":
        return y[-168:].reshape(7, 24).mean(axis=0)
    if model == "ets":
        from statsforecast.models import AutoETS
        return AutoETS(season_length=24).forecast(y=y, h=HORIZON)["mean"]
    if model == "mstl":
        from statsforecast.models import MSTL, AutoETS
        return MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN")).forecast(y=y, h=HORIZON)["mean"]
    if model in ("lgbm", "lgbm_weather"):
        return _lgbm(y, ts, bid, origin, weather=model == "lgbm_weather")
    if model == "automl":
        best, best_mae = None, np.inf
        for m in AUTOML_CANDIDATES:                     # backtest на последних 2 сутках истории
            maes = []
            for k in (2, 1):
                cut = len(y) - k * 24
                p = predict(m, y[:cut], ts[:cut], bid, pd.Timestamp(ts[cut]))
                maes.append(np.mean(np.abs(y[cut:cut + 24] - p)))
            if np.mean(maes) < best_mae:
                best, best_mae = m, np.mean(maes)
        return predict(best, y, ts, bid, origin)
    raise ValueError(model)


def run_window(args):
    (bid, wid), g, models = args
    g = g.sort_values("step")
    hist, fut = g[g.step < 0], g[g.step >= 0]
    y, ts = hist.y_clean.to_numpy(float), pd.DatetimeIndex(hist.timestamp)
    origin, truth = pd.Timestamp(fut.timestamp.iat[0]), fut.y_clean.to_numpy(float)
    assert len(y) == HISTORY and len(truth) == HORIZON
    out = []
    for m in models:
        t0, err = time.time(), None
        try:
            pred = np.asarray(predict(m, y, ts, bid, origin), float)
            if pred.shape != (HORIZON,) or not np.isfinite(pred).all():
                raise ValueError(f"плохой прогноз: shape={pred.shape}")
        except Exception as e:
            pred, err = y[-24:], f"{type(e).__name__}: {e}"
        out.append((m, bid, wid, truth, np.maximum(pred, 0.0), time.time() - t0, err))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    win = pd.read_parquet(WINDOWS[a.split])
    groups = [(k, g, a.models) for k, g in win.groupby(["building_id", "window_id"])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for part in ex.map(run_window, groups, chunksize=4) for r in part]
    total = time.time() - t0

    for m in a.models:
        mr = [r for r in rows if r[0] == m]
        d = RESULTS / f"bl_{m}_{a.split}"
        d.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.DataFrame({"building_id": b, "window_id": w, "hour": range(HORIZON), "y_true": t, "y_pred": p})
                   for _, b, w, t, p, _, _ in mr]).to_csv(d / "predictions.csv", index=False)
        failures = [{"building_id": b, "window_id": int(w), "error": e} for _, b, w, _, _, _, e in mr if e]
        info = {"model": m, "split": a.split, "windows": len(mr), "inputs": "336 h history" + (
                    " + calendar" if m.startswith("lgbm") or m == "automl" else "") + (
                    " + temperature (history actual, forecast day noisy)" if m == "lgbm_weather" else ""),
                "lightgbm_params": LGBM_PARAMS if m.startswith("lgbm") or m == "automl" else None,
                "mean_seconds_per_window": round(float(np.mean([r[5] for r in mr])), 4),
                "failures": failures, "date": datetime.now().isoformat(timespec="seconds"),
                "total_seconds_all_models": round(total, 1)}
        (d / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"bl_{m}_{a.split}: окон {len(mr)}, сбоев {len(failures)}, {info['mean_seconds_per_window']} с/окно")


if __name__ == "__main__":
    main()
