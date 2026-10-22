"""Общие файлы окон для всей команды -> shared/ (хранится в git).

  test_windows.csv / dev_windows.csv — по строке на окно: здание, площадка, тип здания, тип дня,
                                       даты истории и прогноза, флаги in_main_200 / in_repeat_50;
  windows_test.parquet / windows_dev.parquet — сами ряды (история + правильный ответ).
"""
import shutil

import pandas as pd

from one_agent_forecasting.src.config import PROCESSED, ROOT, HISTORY, HORIZON

SHARED = ROOT / "shared"


def spec(win):
    rows = []
    for (b, w), g in win.groupby(["building_id", "window_id"]):
        g = g.sort_values("step")
        ts = g.timestamp
        row = {"building_id": b, "dataset": g.dataset.iat[0], "site": g.site.iat[0], "btype": g.btype.iat[0],
               "window_id": w, "day_type": g.day_type.iat[0],
               "history_start": ts.iat[0], "history_end": ts.iat[HISTORY - 1],
               "forecast_start": ts.iat[HISTORY], "forecast_end": ts.iat[HISTORY + HORIZON - 1],
               "forecast_weekday": ts.iat[HISTORY].day_name()}
        for flag in ("in_main_200", "in_repeat_50"):
            if flag in g:
                row[flag] = bool(g[flag].iat[0])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    SHARED.mkdir(exist_ok=True)
    for split, f in [("test", "windows.parquet"), ("dev", "windows_dev.parquet")]:
        win = pd.read_parquet(PROCESSED / f)
        s = spec(win)
        s.to_csv(SHARED / f"{split}_windows.csv", index=False)
        shutil.copy(PROCESSED / f, SHARED / f"windows_{split}.parquet")
        print(f"{split}: окон {len(s)}, зданий {s.building_id.nunique()}")
        print(s.groupby(["dataset", "day_type"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
