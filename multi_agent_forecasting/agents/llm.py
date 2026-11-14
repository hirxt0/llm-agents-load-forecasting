from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from config import Settings


@dataclass
class Reply:
    message: dict
    calls: list[tuple[str, str, dict]]
    usage: dict


class LLMClient:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise ValueError("Добавь OPENAI_API_KEY в .env")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key, base_url=settings.base_url,
            timeout=180, max_retries=2,
        )

    def chat(self, system: str, messages: list[dict], specs: list[dict]) -> Reply:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            reasoning_effort=self.settings.reasoning_effort,
            max_completion_tokens=self.settings.max_completion_tokens,
            messages=[{"role": "system", "content": system}, *messages],
            tools=[{"type": "function", "function": spec} for spec in specs],
            parallel_tool_calls=False,
        )
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("Expected an object")
            except (ValueError, TypeError):
                args = {"invalid_arguments": True}
            calls.append((call.id, call.function.name, args))
        usage = response.usage
        incoming = getattr(usage, "prompt_tokens_details", None)
        outgoing = getattr(usage, "completion_tokens_details", None)
        return Reply(
            message.model_dump(exclude_none=True), calls,
            {
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
                "tokens_cached": getattr(incoming, "cached_tokens", 0) or 0,
                "tokens_reasoning": getattr(outgoing, "reasoning_tokens", 0) or 0,
                "usage_missing": usage is None,
            },
        )

    def check(self) -> dict:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            reasoning_effort=self.settings.reasoning_effort,
            max_completion_tokens=1024,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        return {
            "requested_model": self.settings.model,
            "returned_model": response.model,
            "answer": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else None,
        }
