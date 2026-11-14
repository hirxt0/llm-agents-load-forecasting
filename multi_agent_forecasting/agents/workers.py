from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Worker:
    description: str
    prompt: str
    tools: tuple[str, ...]


WORKERS = {
    "series_analyst": Worker(
        "Seasonality, recent trend, unusual days and reference 24-hour profiles.",
        "Analyze consumption history. Compare yesterday, last week and typical profiles. "
        "Give a concise recommendation for the next 24 hours with numerical evidence. "
        "Do not clean the already-cleaned history or use observations outside this window.",
        ("describe_series", "get_values", "reference_profiles"),
    ),
    "profile_reviewer": Worker(
        "Independent review of candidate reference profiles and forecast risks.",
        "Independently assess which reference profile is appropriate for the next day. "
        "Challenge unsupported extrapolation of trends, isolated spikes and overly confident changes. "
        "Provide numerical profiles and uncertainty; do not merely repeat the other analyst.",
        ("describe_series", "reference_profiles"),
    ),
    "calendar_analyst": Worker(
        "Calendar, holidays and anonymized building metadata.",
        "Assess how the forecast day differs from historical days by weekday or holiday. "
        "Use only provided calendar and metadata. Recommend justified changes to a reference profile. "
        "Explicitly report unavailable metadata; do not infer it from public datasets.",
        ("calendar", "building_info"),
    ),
    "weather_analyst": Worker(
        "Temperature sensitivity and uncertain day-ahead weather.",
        "Use historical temperature and consumption to assess temperature sensitivity. "
        "Future temperature is a simulated forecast with error. Avoid strong causal claims "
        "based on only 14 days. Recommend only changes supported by tool results.",
        ("weather", "describe_series", "reference_profiles"),
    ),
    "news_analyst": Worker(
        "News revisions available before the forecast origin.",
        "Read only historical news snapshots returned by tools. Never use your memory of events. "
        "Distinguish supported effects from speculation. If no relevant news is found, "
        "recommend no news adjustment.",
        ("news_search", "building_info"),
    ),
}


def available_workers(variant: str) -> list[str]:
    names = ["series_analyst", "profile_reviewer"]
    if variant in ("calendar", "weather", "full"):
        names.append("calendar_analyst")
    if variant in ("weather", "full"):
        names.append("weather_analyst")
    if variant == "full":
        names.append("news_analyst")
    return names


def run_worker(name, task, env, client, ledger, max_steps):
    worker = WORKERS[name]
    permitted = [spec for spec in env.specs() if spec["name"] in worker.tools]
    report_spec = {
        "name": "submit_report",
        "description": "Return your final evidence-based specialist report to the Supervisor.",
        "parameters": {
            "type": "object", "properties": {"report": {"type": "string"}},
            "required": ["report"], "additionalProperties": False,
        },
    }
    messages = [{"role": "user", "content": task}]
    system = (
        env.system_prompt().split("- Typical workflow")[0]
        + "\nYou are a specialist reporting to the Supervisor. Do not submit a final forecast. "
        + worker.prompt
        + f"\nYou have at most {max_steps} model calls, including your final report. "
        "When available, inspect describe_series and reference_profiles first. "
        "Do not request observations older than 336 hours or unavailable context. "
        "Finish with submit_report, returning a concise report in Russian with numerical evidence."
    )
    successful_tools = 0
    for step in range(max_steps):
        if step == max_steps - 1 and successful_tools:
            messages.append({"role": "user", "content": "Tool budget exhausted. Submit your report now using existing evidence."})
            specs = [report_spec]
        else:
            specs = [*permitted, report_spec]
        reply = ledger.chat(client, name, system, messages, specs)
        messages.append(reply.message)
        if not reply.calls:
            if successful_tools and reply.message.get("content"):
                return reply.message["content"]
            messages.append({"role": "user", "content": "Inspect your tools, then report findings."})
            continue
        for call_id, tool_name, args in reply.calls:
            if tool_name == "submit_report":
                report = args.get("report")
                if successful_tools and isinstance(report, str) and report.strip():
                    ledger.tool(name, tool_name, args, json.dumps({"ok": True}))
                    return report
                result = json.dumps({"error": "Inspect a tool and submit a non-empty report"})
            elif tool_name not in {spec["name"] for spec in specs}:
                result = json.dumps({"error": "Tool is not available to this specialist"})
            else:
                result = env.call(tool_name, args)
                if "error" not in json.loads(result):
                    successful_tools += 1
            ledger.tool(name, tool_name, args, result)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    raise RuntimeError(f"{name}: no report within {max_steps} calls")
