from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # one_agent_forecasting/
REPO = ROOT.parent                                   # корень репозитория


def _find_data():
    """Сырые данные не в git: ищем data/ в части агента, в корне репозитория или рядом с ним."""
    for d in (ROOT / "data", REPO / "data", REPO.parent / "data"):
        if (d / "raw").exists():
            return d
    return REPO / "data"


DATA = _find_data()
RAW = DATA / "raw" / "BuildingsBench"
CONTEXT_RAW = DATA / "raw"                 # метаданные BDG-2, погода Лондона
PROCESSED = DATA / "processed"             # промежуточные файлы data.py
CACHE = DATA / "cache"                     # кэш новостей (воспроизводимость и экономия запросов)

SHARED = REPO / "shared"                   # общие для команды окна и инструкции (в git)
RESULTS = REPO / "results"                 # результаты всех систем команды
WINDOWS = {"test": SHARED / "windows_test.parquet",   # единственный источник окон для всех
           "dev": SHARED / "windows_dev.parquet"}

HISTORY = 336
HORIZON = 24
SEED = 42

# цены GPT-5 за 1 млн токенов, USD — сверь с актуальным прайсом перед расчётом
PRICE_PER_M = {"gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
               "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0}}
