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

## Foundation-модели

Zero-shot, без дообучения, локально (CPU/GPU), API не нужен. Запуск из корня репозитория:

```
python -m baselines.run_foundation --split dev --models chronos_bolt --limit 4   # проверка, ничего не пишет
python -m baselines.run_foundation --split test
python -m baselines.run_foundation --split dev
```

| Модель | Что делает |
|---|---|
| `chronos_bolt` | Chronos-Bolt (base), `amazon/chronos-bolt-base`: медиана прогноза, только история |
| `chronos2` | Chronos-2, `amazon/chronos-2`, без ковариат: квантиль 0.5 |
| `chronos2_weather` | То же + температура как ковариата (история - факт, день прогноза - прогноз с той же ошибкой, что у агентов) |
| `timesfm` | TimesFM 2.5 (200M), `google/timesfm-2.5-200m-pytorch`: контекст 336, горизонт 24 |

Правила и формат результатов те же, что у классических моделей. Если модель не загрузилась или упали первые три окна подряд, прогон этой модели останавливается и папку не пишет, чтобы «как вчера» не попал в сводку под её именем. В `run_info.json` дополнительно - чекпойнт, устройство, железо и версии пакетов.

Установка (поверх окружения агента):
```
pip install -r one_agent_forecasting/requirements.txt -r baselines/requirements.txt
git clone https://github.com/google-research/timesfm && pip install -e "timesfm[torch]"   # TimesFM 2.5 нет в PyPI
```
Веса скачиваются с Hugging Face при первом запуске. Для `chronos2_weather` нужны файлы погоды в `data/raw/`, как для `lgbm_weather`.

Видели ли эти модели наши данные при предобучении - [`LEAKAGE.md`](LEAKAGE.md): Moirai и TimesFM 2.5 обучались на корпусах, где есть BDG-2 и LCL, у Chronos-2 - вероятно; их zero-shot результаты на наших зданиях могут быть завышены.
