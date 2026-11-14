from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", encoding="utf-8-sig")


def configured_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else ROOT / path


# Prefer the shared repository data when installed beside one_agent_forecasting.
REPO = ROOT.parent
DATA = next(
    (path for path in (REPO / "one_agent_forecasting" / "data", REPO / "data", REPO.parent / "data")
     if (path / "raw").exists()),
    ROOT / "data",
)
RAW = configured_path("RAW_DIR", str(DATA / "raw" / "BuildingsBench"))
CONTEXT_RAW = configured_path("CONTEXT_DIR", str(DATA / "raw"))
CACHE = ROOT / "data" / "cache"
SHARED = configured_path("WINDOWS_DIR", str(REPO / "shared"))
RESULTS = configured_path("RESULTS_DIR", str(REPO / "results"))


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""), repr=False)
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://apinet.cloud/v1").rstrip("/"))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5"))
    reasoning_effort: str = field(default_factory=lambda: os.getenv("REASONING_EFFORT", "low"))
    max_completion_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_COMPLETION_TOKENS", "8192")))

    def __post_init__(self):
        if self.reasoning_effort not in ("minimal", "low", "medium", "high"):
            raise ValueError("Invalid REASONING_EFFORT")
        if self.max_completion_tokens < 1:
            raise ValueError("MAX_COMPLETION_TOKENS must be positive")

    def cost_usd(self, log: dict):
        if log.get("usage_missing"):
            return None
        values = [os.getenv(name, "") for name in (
            "PRICE_INPUT_PER_M", "PRICE_CACHED_INPUT_PER_M", "PRICE_OUTPUT_PER_M",
        )]
        if not all(values):
            return None
        incoming, cached, outgoing = map(float, values)
        if min(incoming, cached, outgoing) < 0:
            raise ValueError("Token prices must be non-negative")
        return (
            max(0, log["tokens_in"] - log["tokens_cached"]) * incoming
            + log["tokens_cached"] * cached + log["tokens_out"] * outgoing
        ) / 1_000_000
