"""Foundation-модели zero-shot на общих окнах команды (shared/windows_*.parquet).

Запуск из корня репозитория:
    python -m baselines.run_foundation --split dev --models chronos_bolt --limit 4   # проверка, ничего не пишет
    python -m baselines.run_foundation --split test
    python -m baselines.run_foundation --split dev --models timesfm

Модели (имя папки результата: results/bl_<модель>_<split>/):
  chronos_bolt      Chronos-Bolt (base), amazon/chronos-bolt-base: медиана (квантиль 0.5), только история
  chronos2          Chronos-2, amazon/chronos-2, без ковариат: квантиль 0.5
  chronos2_weather  то же + температура как ковариата: история — факт, день прогноза — «прогноз с ошибкой»
                    (та же функция и то же зерно, что у агентов: context.weather(mode="noisy"))
  timesfm           TimesFM 2.5 (200M), google/timesfm-2.5-200m-pytorch: контекст 336, горизонт 24

Правила те же, что в run_baselines.py: модель видит только 336 ч истории своего окна, ничего не
дообучается и не настраивается; отрицательные прогнозы обрезаются до 0; если модель упала на окне —
подставляется snaive_24, и это записывается в run_info.json. Если модель не загрузилась или упали
первые 3 окна подряд, прогон останавливается и папку не пишет, чтобы «как вчера» не попал в сводку
под чужим именем.

Установка (поверх окружения агента):
    pip install -r one_agent_forecasting/requirements.txt -r baselines/requirements.txt
    git clone https://github.com/google-research/timesfm && pip install -e "timesfm[torch]"   # 2.5 нет в PyPI
Веса моделей скачиваются с Hugging Face при первом запуске.
"""
import argparse
import json
import platform
import time
import warnings
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from one_agent_forecasting.src import context
from one_agent_forecasting.src.config import RESULTS, ROOT, HISTORY, HORIZON

try:
    from one_agent_forecasting.src.config import WINDOWS
except ImportError:                                   # config без WINDOWS: общие окна лежат в shared/
    WINDOWS = {s: ROOT.parent / "shared" / f"windows_{s}.parquet" for s in ("test", "dev")}

warnings.filterwarnings("ignore")
MODELS = ["chronos_bolt", "chronos2", "chronos2_weather", "timesfm"]
CHECKPOINTS = {"chronos_bolt": "amazon/chronos-bolt-base", "chronos2": "amazon/chronos-2",
               "chronos2_weather": "amazon/chronos-2", "timesfm": "google/timesfm-2.5-200m-pytorch"}
PACKAGES = {"chronos_bolt": ["torch", "chronos"], "chronos2": ["torch", "chronos"],
            "chronos2_weather": ["torch", "chronos"], "timesfm": ["torch", "timesfm"]}
TIMESFM_CONFIG = dict(max_context=512, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True,
                      force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True)


# ---------------------------------------------------------------- загрузка (один раз на прогон)

def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def _chronos(checkpoint):
    import torch
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(checkpoint, device_map=_device(), torch_dtype=torch.float32)


@lru_cache(maxsize=1)
def _timesfm():
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINTS["timesfm"], torch_compile=False)
    m.compile(timesfm.ForecastConfig(**TIMESFM_CONFIG))
    return m


# ---------------------------------------------------------------- модели

def _chronos2(y, ts, bid, origin, weather):
    """Один ряд в формате predict_df: context — история (+ температура), future_df — только ковариаты."""
    ctx = pd.DataFrame({"id": "w", "timestamp": pd.DatetimeIndex(ts), "target": y})
    fut = None
    if weather:
        idx = pd.date_range(origin, periods=HORIZON, freq="h")
        hist_t, fut_t = context.weather(bid, ts, idx, mode="noisy", key=(bid, str(origin)))
        ctx["temp"] = hist_t.interpolate(limit_direction="both").to_numpy()
        fut = pd.DataFrame({"id": "w", "timestamp": idx, "temp": fut_t})
    pred = _chronos(CHECKPOINTS["chronos2"]).predict_df(
        ctx, future_df=fut, prediction_length=HORIZON, quantile_levels=[0.5],
        id_column="id", timestamp_column="timestamp", target="target")
    return pred.sort_values("timestamp")["0.5" if "0.5" in pred else "predictions"].to_numpy(float)


def predict(model, y, ts, bid, origin):
    if model == "chronos_bolt":
        import torch
        q, _ = _chronos(CHECKPOINTS["chronos_bolt"]).predict_quantiles(
            torch.tensor(y, dtype=torch.float32), prediction_length=HORIZON, quantile_levels=[0.5])
        return q[0, :, 0].cpu().numpy()
    if model in ("chronos2", "chronos2_weather"):
        return _chronos2(y, ts, bid, origin, weather=model == "chronos2_weather")
    if model == "timesfm":
        point, _ = _timesfm().forecast(horizon=HORIZON, inputs=[y])
        return np.asarray(point)[0, :HORIZON]
    raise ValueError(model)


# ---------------------------------------------------------------- прогон

def _versions(names):
    out = {}
    for n in names:
        try:
            out[n] = __import__(n).__version__
        except Exception:
            out[n] = None
    return out


def run_model(m, groups, split, write=True):
    rows, failures, t0 = [], [], time.time()
    for i, ((bid, wid), g) in enumerate(groups):
        g = g.sort_values("step")
        hist, fut = g[g.step < 0], g[g.step >= 0]
        y, ts = hist.y_clean.to_numpy(float), pd.DatetimeIndex(hist.timestamp)
        origin, truth = pd.Timestamp(fut.timestamp.iat[0]), fut.y_clean.to_numpy(float)
        assert len(y) == HISTORY and len(truth) == HORIZON
        t1, err = time.time(), None
        try:
            pred = np.asarray(predict(m, y, ts, bid, origin), float)
            if pred.shape != (HORIZON,) or not np.isfinite(pred).all():
                raise ValueError(f"плохой прогноз: shape={pred.shape}")
        except ImportError:
            raise                                       # нет библиотеки — это не сбой окна
        except Exception as e:
            pred, err = y[-24:], f"{type(e).__name__}: {e}"
            failures.append({"building_id": bid, "window_id": int(wid), "error": err[:300]})
            if len(failures) == i + 1 == 3:             # первые три окна подряд — модель не работает
                raise RuntimeError(f"bl_{m}: первые 3 окна упали, последняя ошибка: {err}") from e
        rows.append((bid, wid, truth, np.maximum(pred, 0.0), time.time() - t1))
        mae = np.mean(np.abs(truth - rows[-1][3]))
        print(f"{bid:32s} w{wid} MAE={mae:.3f} {rows[-1][4]:.2f}s" + (f" FAILED {err}" if err else ""), flush=True)
    total = time.time() - t0
    if not write:
        print(f"bl_{m}_{split}: проверка на {len(rows)} окнах, сбоев {len(failures)} — ничего не записано")
        return

    d = RESULTS / f"bl_{m}_{split}"
    d.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.DataFrame({"building_id": b, "window_id": w, "hour": range(HORIZON), "y_true": t, "y_pred": p})
               for b, w, t, p, _ in rows]).to_csv(d / "predictions.csv", index=False)
    info = {"model": m, "checkpoint": CHECKPOINTS[m], "split": split, "windows": len(rows),
            "inputs": "336 h history" + (" + temperature (history actual, forecast day noisy)"
                                         if m == "chronos2_weather" else ""),
            "params": {"mode": "zero-shot", "output": "median (quantile 0.5)" if m != "timesfm" else "point forecast",
                       "prediction_length": HORIZON, **(TIMESFM_CONFIG if m == "timesfm" else {}),
                       "device": _device()},
            "mean_seconds_per_window": round(float(np.mean([r[4] for r in rows])), 4),
            "failures": failures, "date": datetime.now().isoformat(timespec="seconds"),
            "total_seconds": round(total, 1),
            "hardware": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                         "python": platform.python_version()},
            "packages": _versions(["numpy", "pandas", *PACKAGES[m]])}
    (d / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bl_{m}_{split}: окон {len(rows)}, сбоев {len(failures)}, {info['mean_seconds_per_window']} с/окно")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--limit", type=int, default=None, help="проверка на первых N окнах; результаты не пишутся")
    a = ap.parse_args()

    win = pd.read_parquet(WINDOWS[a.split])
    groups = list(win.groupby(["building_id", "window_id"]))[: a.limit]
    for m in a.models:
        try:
            run_model(m, groups, a.split, write=a.limit is None)
        except Exception as e:                          # одна модель не должна останавливать остальные
            print(f"\n!!! bl_{m}_{a.split} пропущен: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
