import json
import os
import time
from openai import OpenAI


SYSTEM_PROMPT = """You are a data analyst agent. You receive the hourly electricity consumption history \
(kWh) of one building: {length} hours ending at {end}. Your goal is the most accurate forecast \
of the next 24 hours.

The history may contain data defects typical for smart meters: missing values, communication gaps, \
spikes or drops, and stuck-meter segments (constant values or zeros). Defects in the history hurt \
the forecast, but genuine consumption patterns (nights, weekends, holidays) must NOT be removed.

Work step by step with the tools:
1. Inspect the series and decide whether and how to clean it.
2. Make sure there are no missing values left (impute) before backtesting or forecasting.
3. Compare candidate models / hyperparameters with backtest (limited budget).
4. Finish with submit_forecast. You must always call submit_forecast in the end.
You never see the future values; do not guess them, rely on tools."""

REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _load_dotenv():
    """Ключи из файла .env в корне репозитория (он в .gitignore и в git не попадает)."""
    from pathlib import Path
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class LLMClient:
    def __init__(self, provider, model, base_url=None, temperature=0.0, reasoning_effort="low"):
        self.provider, self.model = provider, model
        self.temperature, self.reasoning_effort = temperature, reasoning_effort
        self.is_reasoning = model.startswith(REASONING_PREFIXES)
        self.client = OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                                api_key=os.getenv("OPENAI_API_KEY", "ollama"), max_retries=5, timeout=300)

    def tools_format(self, specs):
        if self.provider == "anthropic":
            return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                    for t in specs]
        return [{"type": "function", "function": t} for t in specs]

    def chat(self, system, messages, specs):
        """Возвращает (текст, [(id, имя, аргументы)], сообщение ассистента, usage-словарь)."""
        if self.provider == "anthropic":
            r = self.client.messages.create(model=self.model, system=system, messages=messages,
                                            tools=self.tools_format(specs), max_tokens=4096,
                                            temperature=self.temperature)
            calls = [(b.id, b.name, b.input) for b in r.content if b.type == "tool_use"]
            text = "".join(b.text for b in r.content if b.type == "text")
            msg = {"role": "assistant", "content": [b.model_dump() for b in r.content]}
            return text, calls, msg, {"in": r.usage.input_tokens, "out": r.usage.output_tokens,
                                      "reasoning": 0, "cached": 0}
        kw = dict(model=self.model, messages=[{"role": "system", "content": system}] + messages,
                  tools=self.tools_format(specs))
        if self.is_reasoning:      # у GPT-5 нет temperature, зато есть уровень рассуждений
            kw["reasoning_effort"] = self.reasoning_effort
        else:
            kw["temperature"] = self.temperature
        r = self.client.chat.completions.create(**kw)
        m = r.choices[0].message
        calls = []
        for c in m.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": c.function.arguments}
            calls.append((c.id, c.function.name, args))
        msg = {"role": "assistant", "content": m.content or "",
               **({"tool_calls": [c.model_dump() for c in m.tool_calls]} if m.tool_calls else {})}
        u = r.usage
        det_out = getattr(u, "completion_tokens_details", None)
        det_in = getattr(u, "prompt_tokens_details", None)
        usage = {"in": u.prompt_tokens if u else 0, "out": u.completion_tokens if u else 0,
                 "reasoning": getattr(det_out, "reasoning_tokens", 0) or 0,
                 "cached": getattr(det_in, "cached_tokens", 0) or 0}
        return m.content or "", calls, msg, usage

    def tool_results(self, results):
        """results: [(id, текст)] → сообщения с результатами инструментов."""
        if self.provider == "anthropic":
            return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": c}
                                                 for i, c in results]}]
        return [{"role": "tool", "tool_call_id": i, "content": c} for i, c in results]


def run_agent(env, llm, max_steps=20, system=None):
    """Прогоняет агента на одном окне. Возвращает лог (токены, время, вызовы, сбой) и диалог."""
    if system is None:
        system = env.system_prompt() if hasattr(env, "system_prompt") else \
            SYSTEM_PROMPT.format(length=len(env.y), end=env.ts[-1])
    first = env.initial_message() if hasattr(env, "initial_message") else "Start. Produce the 24-hour forecast."
    messages = [{"role": "user", "content": first}]
    log = {"tokens_in": 0, "tokens_out": 0, "tokens_reasoning": 0, "tokens_cached": 0,
           "llm_calls": 0, "tool_calls": 0, "error": None}
    t0 = time.time()
    try:
        for _ in range(max_steps):
            text, calls, msg, usage = llm.chat(system, messages, env.specs())
            log["llm_calls"] += 1
            log["tokens_in"] += usage["in"]
            log["tokens_out"] += usage["out"]
            log["tokens_reasoning"] += usage["reasoning"]
            log["tokens_cached"] += usage["cached"]
            messages.append(msg)
            if not calls:
                messages.append({"role": "user", "content": "Use the tools. Finish with submit_forecast."})
                continue
            results = []
            for cid, name, args in calls:
                log["tool_calls"] += 1
                results.append((cid, env.call(name, args)))
            messages += llm.tool_results(results)
            if env.done:
                break
    except Exception as e:
        log["error"] = f"{type(e).__name__}: {e}"
    log["seconds"] = round(time.time() - t0, 2)
    log["failed"] = not env.done
    log["messages"] = messages
    return log
