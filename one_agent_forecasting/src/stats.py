"""Статистика сравнения методов: единица анализа — ЗДАНИЕ, а не окно.

Окна одного здания зависимы, поэтому для каждого здания сначала усредняется
лог-отношение ошибок двух методов, а уже потом идёт тест/интервал по зданиям.

  paired_compare(a, b) — для пар методов на одних и тех же окнах:
      ratio  — геометрическое среднее отношение NRMSE(b)/NRMSE(a) по зданиям (< 1 — b лучше);
      ci     — 95% кластерный бутстреп по зданиям;
      p      — Wilcoxon по средним лог-отношениям зданий;
  holm(pvalues)          — поправка Холма на семейство сравнений;
  equivalent(ratio_ci90, margin) — тест эквивалентности (TOST через 90% ДИ).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

B = 2000
RNG = np.random.default_rng(0)


def building_log_ratios(w, a, b, metric="NRMSE"):
    """Среднее по окнам log(metric_b / metric_a) для каждого здания (только общие окна)."""
    p = w[w.method.isin([a, b])].pivot_table(index=["building_id", "window_id"], columns="method",
                                             values=metric).dropna()
    if a not in p or b not in p:                    # у одного из методов нет окон в этой выборке
        return pd.Series(dtype=float), 0
    p = p[(p[a] > 0) & (p[b] > 0)]
    lr = np.log(p[b] / p[a])
    return lr.groupby(level="building_id").mean(), len(p)


def bootstrap_ci(x, level=0.95):
    x = np.asarray(x, float)
    means = x[RNG.integers(0, len(x), (B, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(np.exp(lo)), float(np.exp(hi))


def paired_compare(w, a, b, metric="NRMSE"):
    lr, n_win = building_log_ratios(w, a, b, metric)
    if len(lr) < 5:
        return None
    p = wilcoxon(lr).pvalue if (lr != 0).any() else 1.0
    return {"a": a, "b": b, "n_buildings": len(lr), "n_windows": n_win,
            "ratio": float(np.exp(lr.mean())), "ci95": bootstrap_ci(lr, 0.95),
            "ci90": bootstrap_ci(lr, 0.90), "p": float(p)}


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def equivalent(ci90, margin=0.05):
    """Эквивалентны ли методы с точностью ±margin (TOST: 90% ДИ отношения внутри границ)."""
    return ci90[0] > 1 - margin and ci90[1] < 1 + margin
