from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# данные могут лежать внутри репозитория или рядом с ним (../data)
DATA = ROOT / "data" if (ROOT / "data" / "raw").exists() else ROOT.parent / "data"
RAW = DATA / "raw" / "BuildingsBench"
CONTEXT_RAW = DATA / "raw"                 # метаданные BDG-2, погода Лондона
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"                     # кэш новостей (воспроизводимость и экономия запросов)
RESULTS = ROOT / "results"

HISTORY = 336
HORIZON = 24
WINDOWS_PER_BUILDING = 8
N_COMMERCIAL = 25
N_RESIDENTIAL = 25
SEED = 42

LEVELS = ["clean", "low", "medium", "high"]

# отладочная выборка: здания, которых нет в тесте (на них настраиваем промпт)
DEV_COMMERCIAL = 5
DEV_RESIDENTIAL = 5
DEV_WINDOWS_PER_BUILDING = 4

# цены GPT-5 за 1 млн токенов, USD — сверь с актуальным прайсом OpenAI перед расчётом
PRICE_PER_M = {"gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
               "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0}}
