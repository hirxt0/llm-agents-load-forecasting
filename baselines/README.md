# Бейзлайны

Классические модели на общих окнах команды (`shared/windows_*.parquet`). Запуск из корня репозитория:

```
python -m baselines.run_baselines --split test      # все 400 тестовых окон
python -m baselines.run_baselines --split dev       # 40 отладочных окон
```

| Модель | Что делает |
|---|---|
| `snaive_24` | Как вчера |
| `snaive_168` | Как неделю назад |
| `persistence_ens` | Persistence Ensemble из BuildingsBench (`AveragePersistence`): среднее того же часа за 7 предыдущих суток |
| `ets` | AutoETS, сезон 24 |
| `mstl` | MSTL, сезоны 24 и 168 |
| `lgbm` | LightGBM по окну: лаги 24–168, час, день недели, праздник |
| `lgbm_weather` | То же + температура (история - факт, день прогноза - прогноз с той же ошибкой, что у агентов) |
| `automl` | Лучшая из классических моделей по backtest на последних 2 сутках истории |

Каждая модель видит только 336 часов истории своего окна, гиперпараметры фиксированы. Результаты - `results/bl_<модель>_<split>/` (`predictions.csv` и `run_info.json`), сводка - `python -m one_agent_forecasting.src.summarize`.

Foundation-модели (Chronos-Bolt, Chronos-2, TimesFM) и проверка их обучающих корпусов - по ТЗ в `shared/BASELINE_TZ.md`, отдельным файлом в этой папке.
