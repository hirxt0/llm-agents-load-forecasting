from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from agents.workers import WORKERS, available_workers, run_worker


@dataclass
class Ledger:
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    tokens_reasoning: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    usage_missing: bool = False
    events: list[dict] = field(default_factory=list)

    def chat(self, client, agent, system, messages, specs):
        self.llm_calls += 1
        try:
            reply = client.chat(system, messages, specs)
        except Exception:
            self.usage_missing = True
            raise
        for key in ("tokens_in", "tokens_out", "tokens_cached", "tokens_reasoning"):
            setattr(self, key, getattr(self, key) + reply.usage[key])
        self.usage_missing |= reply.usage.get("usage_missing", False)
        self.events.append({
            "agent": agent, "system": system,
            "messages": [*messages, reply.message], "usage": reply.usage,
        })
        return reply

    def tool(self, agent, name, args, result):
        self.tool_calls += 1
        self.events.append({"agent": agent, "tool": name, "args": args, "result": result})

    def counts(self):
        return {key: getattr(self, key) for key in (
            "tokens_in", "tokens_out", "tokens_cached", "tokens_reasoning",
            "llm_calls", "tool_calls", "usage_missing",
        )}


def run_supervisor(env, client, max_steps=8, worker_steps=5):
    ledger = Ledger()
    started = time.monotonic()
    names = available_workers(env.variant)
    reports = {}
    tools = [
        {
            "name": "delegate",
            "description": "Assign a specific analysis task to a specialist; receive their report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": names},
                    "task": {"type": "string"},
                },
                "required": ["agent", "task"], "additionalProperties": False,
            },
        },
        next(spec for spec in env.specs() if spec["name"] == "submit_forecast"),
    ]
    team = "\n".join(f"{name}: {WORKERS[name].description}" for name in names)
    system = (
        env.system_prompt()
        + "\nYou are the Supervisor of a forecasting team:\n" + team
        + "\nDelegate to EVERY available specialist at least once before submission. "
        "Send a specific task. Specialists inspect only this window's history and allowed context. "
        "Synthesize their reports into your own forecast: exactly 24 finite non-negative kWh values "
        "for hours 00-23. Resolve disagreements using evidence. Do not mechanically average reports. "
        "Use submit_forecast to finish. Give a short justification in Russian."
    )
    messages = [{"role": "user", "content": "Analyze this window and submit the day-ahead forecast."}]
    error = None
    try:
        for _ in range(max_steps):
            reply = ledger.chat(client, "supervisor", system, messages, tools)
            messages.append(reply.message)
            if not reply.calls:
                messages.append({"role": "user", "content": "Use delegate or submit_forecast to continue."})
                continue
            for call_id, name, args in reply.calls:
                try:
                    if name == "delegate":
                        worker = args.get("agent")
                        if worker not in names:
                            raise ValueError("Unknown or disabled specialist")
                        if worker in reports:
                            result = json.dumps({"report": reports[worker], "cached": True}, ensure_ascii=False)
                        else:
                            task = args.get("task")
                            if not isinstance(task, str) or not task.strip():
                                raise ValueError("A non-empty task is required")
                            print(f"  supervisor -> {worker}", flush=True)
                            reports[worker] = run_worker(worker, task, env, client, ledger, worker_steps)
                            result = json.dumps({"report": reports[worker]}, ensure_ascii=False)
                    elif name == "submit_forecast":
                        missing = sorted(set(names) - reports.keys())
                        if missing:
                            result = json.dumps({"error": "Missing specialist reports", "agents": missing})
                        else:
                            result = env.call(name, args)
                    else:
                        raise ValueError("Unknown tool")
                except (ValueError, TypeError, KeyError) as exc:
                    result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                ledger.tool("supervisor", name, args, result)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
                if env.done:
                    break
            if env.done:
                break
        if not env.done:
            error = "Supervisor exceeded the call limit without a valid forecast"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        **ledger.counts(), "seconds": round(time.monotonic() - started, 3),
        "failed": not env.done, "error": error, "reports": reports, "events": ledger.events,
    }
