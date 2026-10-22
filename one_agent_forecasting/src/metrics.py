"""Метрики качества прогноза — общие для всей команды.

Основная метрика — NRMSE (= CV(RMSE), как в BuildingsBench), нормированная по среднему
ИСТОРИИ окна (336 ч), а не прогнозируемых суток: так знаменатель не проваливается в дни
низкого потребления и одинаков для всех методов.

  NRMSE, % — RMSE / mean(history) · 100          (основная)
  NMAE,  % — MAE  / mean(history) · 100
  NMBE,  % — mean(y_pred − y_true) / mean(history) · 100   (> 0 — завышение, < 0 — занижение)
  MASE     — MAE / средняя |y_t − y_{t−24}| на истории    (< 1 — лучше прогноза «как вчера»)
  MAE, RMSE — в кВт·ч (только внутри одного датасета: зависят от масштаба здания)
  MAPE,  % — вторичная, только для коммерческих зданий (BDG-2): у домов LCL потребление
             бывает около нуля, и MAPE не определена.
"""
import numpy as np


def forecast_metrics(y_true, y_pred, history):
    """Метрики одного прогноза на 24 часа. history — 336 часов истории этого окна."""
    y_true, y_pred, h = (np.asarray(a, float) for a in (y_true, y_pred, history))
    err = y_pred - y_true
    scale = h.mean()
    naive = np.mean(np.abs(h[24:] - h[:-24]))
    mae, rmse = np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2))
    return {
        "NRMSE": float(rmse / scale * 100),
        "NMAE": float(mae / scale * 100),
        "NMBE": float(err.mean() / scale * 100),
        "MASE": float(mae / naive) if naive > 0 else float("nan"),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100),
    }


def cleaning_metrics(label, flagged, y_cleaned, y_clean):
    """Метрики очистки (первая версия эксперимента с грязными данными)."""
    defect = (label == 2) | (label == 3)
    pred = np.zeros(len(label), bool)
    if len(flagged):
        pred[np.asarray(flagged, int)] = True
    pred &= label != 1
    tp = int((pred & defect).sum())
    precision = tp / pred.sum() if pred.sum() else np.nan
    recall = tp / defect.sum() if defect.sum() else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision + recall) else np.nan)
    bad = label > 0
    nmae = (np.mean(np.abs(y_cleaned[bad] - y_clean[bad])) / (np.mean(y_clean) + 1e-9) * 100
            if bad.any() else np.nan)
    return {"det_precision": precision, "det_recall": recall, "det_f1": f1,
            "repair_NMAE": nmae, "n_defects": int(bad.sum())}
