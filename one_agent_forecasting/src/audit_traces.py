"""Аудит рассуждений агента: упоминает ли он факты, которых не было в ответах инструментов.

Для каждого прогона ищем в тексте агента (сообщения + обоснование прогноза):
  * годы и названия событий/праздников, которых нет в выдаче инструментов;
  * «подозрительные» слова, говорящие о внешнем знании (hurricane, blackout, Olympics …).
Каждая находка — кандидат на утечку через память модели; их долю указываем в статье,
а сами случаи можно разобрать вручную.

  python audit_traces.py ../results/fc_gpt-5-low_full_noisy_test
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

SUSPICIOUS = ["olympic", "hurricane", "storm", "blackout", "outage", "heatwave", "heat wave", "snow",
              "flood", "strike", "festival", "concert", "match", "election", "brexit", "jubilee",
              "christmas", "thanksgiving", "easter", "new year", "spring break", "semester", "exam",
              "graduation", "covid", "lockdown", "wimbledon", "super bowl", "eclipse"]


def texts(rec):
    agent, tools = [], []
    for m in rec.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):          # формат Anthropic
            for b in content:
                if b.get("type") == "text":
                    agent.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    agent.append(json.dumps(b.get("input", {})))
                elif b.get("type") == "tool_result":
                    tools.append(str(b.get("content", "")))
            continue
        if m.get("role") == "assistant":
            agent.append(content or "")
            for c in m.get("tool_calls", []) or []:
                agent.append(c.get("function", {}).get("arguments", ""))
        elif m.get("role") in ("tool", "user"):
            tools.append(content or "")
    agent.append(rec.get("reasoning", ""))
    return " ".join(agent).lower(), " ".join(tools).lower()


def audit(run_dir):
    rows = []
    for line in open(Path(run_dir) / "trajectories.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        agent, tools = texts(rec)
        years = {y for y in re.findall(r"\b(?:19|20)\d{2}\b", agent) if y not in tools}
        words = {w for w in SUSPICIOUS if w in agent and w not in tools}
        rows.append({"building_id": rec["building_id"], "window_id": rec["window_id"],
                     "unsupported_years": " ".join(sorted(years)), "unsupported_terms": " ".join(sorted(words)),
                     "flagged": bool(years or words)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(run_dir) / "audit.csv", index=False)
    print(f"{run_dir}: прогонов {len(df)}, с подозрительными упоминаниями {df.flagged.mean():.1%}")
    if df.flagged.any():
        print(df[df.flagged].head(15).to_string(index=False))
    return df


if __name__ == "__main__":
    for d in sys.argv[1:]:
        audit(d)
