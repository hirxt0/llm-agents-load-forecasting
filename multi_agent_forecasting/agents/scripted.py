from __future__ import annotations

import json

from agents.llm import Reply
from agents.workers import available_workers


class ScriptedClient:
    def __init__(self, variant):
        self.names = available_workers(variant)

    def chat(self, system, messages, specs):
        tools = {spec["name"] for spec in specs}
        if "delegate" in tools:
            delegated = [
                call["function"]["name"] for message in messages
                for call in message.get("tool_calls", [])
            ].count("delegate")
            if delegated < len(self.names):
                name, args = "delegate", {"agent": self.names[delegated], "task": "Inspect all your tools and report evidence."}
            else:
                reports = [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]
                profile = next(json.loads(report["report"])["profile"] for report in reports if "report" in report and "profile" in json.loads(report["report"]))
                name, args = "submit_forecast", {"values": profile, "reasoning": "Offline scripted check; not an LLM forecast."}
        else:
            used = {
                call["function"]["name"] for message in messages
                for call in message.get("tool_calls", [])
            }
            needed = [spec["name"] for spec in specs if spec["name"] not in ("get_values", "submit_report")]
            remaining = [name for name in needed if name not in used]
            if remaining:
                name = remaining[0]
                args = {"date": "2000-01-01"} if name == "news_search" else {}
            else:
                report = {"findings": "Offline tools inspected"}
                for message in messages:
                    if message.get("role") == "tool":
                        result = json.loads(message["content"])
                        if isinstance(result, dict) and "mean_same_weekday_last_2_weeks" in result:
                            report["profile"] = result["mean_same_weekday_last_2_weeks"]
                return Reply({"role": "assistant", "content": json.dumps(report)}, [], self.usage())
        call_id = f"offline-{len(messages)}"
        return Reply(
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }]},
            [(call_id, name, args)], self.usage(),
        )

    @staticmethod
    def usage():
        return {"tokens_in": 0, "tokens_out": 0, "tokens_cached": 0, "tokens_reasoning": 0, "usage_missing": False}
