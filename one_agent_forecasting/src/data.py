"""Отбор зданий и окон (версия 2: гигиена данных + стратификация по типам дней).

Правила отбора ЗДАНИЯ (один год на здание):
  * нет пропусков, среднее > 0, нулевых часов < 1%, залипаний (одно значение подряд) не дольше 12 ч;
  * доля «заполненных» часов < 1%: участки, где 4+ точки лежат на одной прямой (след
    линейной интерполяции при очистке BDG-2/LCL) или значение не меняется 4+ часа;
  * жилые (LCL): полный год (≥ 8000 ч) и среднее ≥ 0.05 кВт·ч.
Каждое здание — 4 окна, по одному на тип прогнозируемых суток:
  weekday  — обычный будний день;   weekend — обычный выходной;
  holiday  — государственный праздник (США: федеральные + штата, Великобритания: Англия);
  weather  — погодная аномалия: среднесуточная температура сильнее всего отклоняется от
             31-дневной скользящей средней (верхние 5% модуля отклонения за год).
Правила ОКНА: 336 ч истории + 24 ч прогноза с полуночи; окна одного здания не пересекаются;
в прогнозируемых сутках нет заполненных часов; в истории их ≤ 2%.

Выборки (здания не пересекаются):
  test — 50 коммерческих + 50 жилых зданий × 4 окна = 400 окон;
         main_200 — по 2 окна на здание: обычный день (будний/выходной поочерёдно)
                    и особый (праздник/погода поочерёдно) → по 50 окон каждого типа;
  dev  — 5 + 5 других зданий × 4 окна = 40 окон (только для настройки).

Время: BDG-2 — постоянное местное время без перехода на летнее, LCL — GMT.
"""
import json

import holidays
import numpy as np
import pandas as pd

from one_agent_forecasting.src.config import RAW, CONTEXT_RAW, PROCESSED, HISTORY, HORIZON, SEED

N_TEST, N_DEV = 50, 5                      # зданий на датасет
DAY_TYPES = ["weekday", "weekend", "holiday", "weather"]
SITE_STATE = {"Bear": "CA", "Fox": "AZ", "Panther": "FL", "Rat": "DC"}


# ---------------------------------------------------------------- качество ряда

def _runs(mask):
    out, s = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    return out


def imputed_mask(v):
    """Часы, похожие на заполненные: 4+ точек на прямой (в т.ч. 4+ одинаковых значения)."""
    d = np.diff(v)
    same_step = np.r_[False, np.isclose(d[1:], d[:-1], atol=1e-6, rtol=0)]
    m = np.zeros(len(v), bool)
    for s, e in _runs(same_step):
        if e - s >= 2:
            m[s - 1:e + 1] = True
    return m


def _longest(mask):
    return max((e - s for s, e in _runs(mask)), default=0)


def series_ok(v, residential):
    if np.isnan(v).any() or v.mean() <= 0 or (v == 0).mean() >= 0.01:
        return False
    if _longest(np.r_[False, np.diff(v) == 0]) > 12 or imputed_mask(v).mean() >= 0.01:
        return False
    return not residential or (len(v) >= 8000 and v.mean() >= 0.05)


# ---------------------------------------------------------------- погода и календарь

def _weather(site):
    if site == "LCL":
        j = json.load(open(CONTEXT_RAW / "weather_london_openmeteo.json"))
        s = pd.Series(j["hourly"]["temperature_2m"], index=pd.to_datetime(j["hourly"]["time"]))
    else:
        s = pd.read_csv(RAW / "BDG-2" / f"weather_{site}.csv", index_col=0, parse_dates=True)["temperature"]
    s = s[~s.index.duplicated()].sort_index().resample("h").mean().interpolate(limit=6)
    daily = s.resample("D").mean()
    return (daily - daily.rolling(31, center=True, min_periods=15).mean()).abs()


def _holidays(site, years):
    if site == "LCL":
        return holidays.country_holidays("GB", subdiv="ENG", years=years)
    return holidays.country_holidays("US", subdiv=SITE_STATE[site], years=years)


# ---------------------------------------------------------------- кандидаты

def load_candidates():
    rows = []
    for f in sorted((RAW / "BDG-2").glob("*_clean=*.csv")):
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        for c in d.columns:
            if series_ok(d[c].to_numpy(float), residential=False):
                rows.append(dict(building_id=f"{c}@{f.stem.split('=')[1]}", name=c, dataset="BDG-2",
                                 btype=c.split("_")[1], site=c.split("_")[0], series=d[c]))
    for f in sorted((RAW / "LCL").glob("*_clean=*.csv")):
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        if series_ok(s.to_numpy(float), residential=True):
            name = f.stem.split("_clean")[0]
            rows.append(dict(building_id=f"{name}@{f.stem.split('=')[1]}", name=name, dataset="LCL",
                             btype="residential", site="LCL", series=s))
    return pd.DataFrame(rows)


def pick_buildings(cand, rng):
    """Одно здание — один год; коммерческие стратифицированы по площадке и типу здания."""
    cand = cand.sample(frac=1, random_state=SEED).drop_duplicates("name")
    com = cand[cand.dataset == "BDG-2"]
    # поочерёдно берём по площадкам, внутри площадки — разные типы, чтобы не было 40 учебных корпусов
    com = com.assign(k=com.groupby(["site", "btype"]).cumcount()).sort_values(["k", "site"])
    com = com.groupby("site", group_keys=False).head(10_000)
    res = cand[cand.dataset == "LCL"]
    n = N_TEST + N_DEV
    return com.head(n).reset_index(drop=True), res.head(n).reset_index(drop=True)


# ---------------------------------------------------------------- окна

def day_types(b):
    s = b["series"]
    days = pd.DatetimeIndex(sorted(set(s.index.normalize())))
    hol = _holidays(b["site"], sorted(set(days.year)))
    anom = _weather(b["site"]).reindex(days)
    thr = anom.quantile(0.95)
    out = {}
    for d in days:
        if d.date() in hol:
            out[d] = "holiday"
        elif anom.get(d, np.nan) >= thr:
            out[d] = "weather"
        elif d.dayofweek >= 5:
            out[d] = "weekend"
        else:
            out[d] = "weekday"
    return out


def make_windows(b, rng):
    s = b["series"]
    v = s.to_numpy(float)
    imp = imputed_mask(v)
    pos = {t: i for i, t in enumerate(s.index)}
    types = day_types(b)
    cands = {t: [] for t in DAY_TYPES}
    for d, t in types.items():
        o = pos.get(d)
        if o is None or o < HISTORY or o + HORIZON > len(v):
            continue
        if imp[o:o + HORIZON].any() or imp[o - HISTORY:o].mean() > 0.02:
            continue
        cands[t].append(o)
    chosen = {}
    for t in ["holiday", "weather", "weekend", "weekday"]:      # редкие типы выбираем первыми
        for o in rng.permutation(cands[t]):
            if all(abs(o - c) >= HISTORY + HORIZON for c in chosen.values()):
                chosen[t] = int(o)
                break
    if len(chosen) < 4:
        return None
    out = []
    for w, t in enumerate(DAY_TYPES):
        o = chosen[t]
        sl = slice(o - HISTORY, o + HORIZON)
        out.append(pd.DataFrame({
            "building_id": b["building_id"], "dataset": b["dataset"], "btype": b["btype"],
            "site": b["site"], "window_id": w, "day_type": t, "step": np.arange(-HISTORY, HORIZON),
            "timestamp": s.index[sl], "y_clean": v[sl]}))
    return out


def build(buildings, n, rng):
    parts, used = [], []
    for _, b in buildings.iterrows():
        w = make_windows(b, rng)
        if w is not None:
            parts += w
            used.append(b["building_id"])
        if len(used) == n:
            break
    return parts, used


def main():
    rng = np.random.default_rng(SEED)
    cand = load_candidates()
    print("прошли фильтр качества:", cand.groupby("dataset").size().to_dict())
    com, res = pick_buildings(cand, rng)
    test, dev = [], []
    for pool in (com, res):
        p, used = build(pool, N_TEST, rng)
        test += p
        rest = pool[~pool.building_id.isin(used)]
        p, _ = build(rest, N_DEV, rng)
        dev += p
    test, dev = pd.concat(test, ignore_index=True), pd.concat(dev, ignore_index=True)
    assert not set(test.building_id) & set(dev.building_id)

    # основная выборка: обычный день (будний/выходной поочерёдно) + особый (праздник/погода поочерёдно)
    blds = test.drop_duplicates("building_id")[["building_id", "dataset"]].reset_index(drop=True)
    blds["k"] = blds.groupby("dataset").cumcount()
    regular = np.where(blds.k % 2 == 0, "weekday", "weekend")
    special = np.where((blds.k // 2) % 2 == 0, "holiday", "weather")
    main_pairs = set(zip(blds.building_id, regular)) | set(zip(blds.building_id, special))
    test["in_main_200"] = [(b, t) in main_pairs for b, t in zip(test.building_id, test.day_type)]
    # повтор: 50 окон из основных, поровну по датасетам и типам дней
    m = test[test.in_main_200].drop_duplicates(["building_id", "window_id"])
    rep = m.groupby(["dataset", "day_type"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), 7 if g.name[1] == "holiday" else 6), random_state=SEED))
    rep_keys = set(zip(m.loc[rep.index, "building_id"], m.loc[rep.index, "window_id"]))
    test["in_repeat_50"] = [(b, w) in rep_keys for b, w in zip(test.building_id, test.window_id)]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    test.to_parquet(PROCESSED / "windows.parquet", index=False)
    dev.to_parquet(PROCESSED / "windows_dev.parquet", index=False)
    for name, df in [("buildings.csv", test), ("buildings_dev.csv", dev)]:
        df.drop_duplicates("building_id")[["building_id", "dataset", "btype", "site"]].to_csv(PROCESSED / name, index=False)

    fc = test[test.step == 0]
    print("тест:", fc.building_id.nunique(), "зданий,", len(fc), "окон | dev:", dev[dev.step == 0].building_id.nunique(),
          "зданий,", len(dev[dev.step == 0]), "окон")
    print("основные 200 по типам дней:", fc[fc.in_main_200].groupby(["dataset", "day_type"]).size().unstack().to_string())
    print("повтор:", int(fc.in_repeat_50.sum()), "| коммерческие по площадкам и типам:")
    print(fc[fc.dataset == "BDG-2"].drop_duplicates("building_id").groupby(["site", "btype"]).size().unstack(fill_value=0).to_string())
    print("прогнозы:", fc.groupby("dataset").timestamp.agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
