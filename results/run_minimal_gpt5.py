"""Run the agreed minimal GPT-5 experiment and keep a usage ledger."""
import csv
import json
import msvcrt
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from one_agent_forecasting.src import audit_traces, probe_memorization as probe
from one_agent_forecasting.src import run_forecast_agent as runner, summarize
from one_agent_forecasting.src.agent import LLMClient


MODEL = "gpt-5"
EFFORT = "low"
LIMIT = 39.0
VARIANTS = ("no_tools", "series", "calendar", "weather", "full")
LEDGER = runner.RESULTS / "gpt5_minimal_usage.jsonl"
STATUS = runner.RESULTS / "gpt5_minimal_status.json"
LOCK = threading.Lock()
LOCAL = threading.local()
PHASE = ""
SPENT = 0.0
STOP = ""
ORIGINAL_CHAT = LLMClient.chat


class RunStopped(BaseException):
    pass


def status(state, reason=""):
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(state=state, phase=PHASE, estimated_usd=SPENT,
                                    budget_usd=LIMIT, reason=reason), indent=2))
    temp.replace(STATUS)


def metered_chat(self, system, messages, specs):
    global SPENT, STOP
    with LOCK:
        if STOP:
            raise RunStopped(STOP)
        if SPENT >= LIMIT:
            STOP = "Estimated budget reached. In-flight requests can add to the total."
            raise RunStopped(STOP)
    try:
        result = ORIGINAL_CHAT(self, system, messages, specs)
    except Exception as exc:
        with LOCK:
            STOP = f"Provider call failed ({type(exc).__name__}, HTTP {getattr(exc, 'status_code', '?')})."
        raise RunStopped(STOP) from None
    usage = result[3]
    cost = runner.cost_usd(MODEL, {f"tokens_{k}": v for k, v in usage.items()})
    with LOCK:
        SPENT += cost
        row = dict(phase=PHASE, window=getattr(LOCAL, "window", None), **usage, estimated_usd=cost)
        with LEDGER.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if usage["in"] + usage["out"] == 0:
            STOP = "The provider returned no token usage."
            raise RunStopped(STOP)
    if hasattr(LOCAL, "usage"):
        for key, value in usage.items():
            LOCAL.usage[key] += value
        LOCAL.usage["llm_calls"] += 1
    return result


def forecast(split, variant, *extra):
    global PHASE
    PHASE = " ".join((split, variant, *extra))
    status("running")
    sys.argv = ["run_forecast_agent", "--model", MODEL, "--reasoning-effort", EFFORT,
                "--split", split, "--variant", variant, "--workers", "4", *extra]
    runner.main()


def probe_case(case, out):
    (bid, wid), g, identity = case
    hist, fut = g[g.step < 0], g[g.step >= 0]
    dataset = g.dataset.iat[0]
    LOCAL.window = [bid, int(wid), identity]
    LOCAL.usage = dict.fromkeys(("in", "out", "reasoning", "cached", "llm_calls"), 0)
    client = LLMClient("openai", MODEL, reasoning_effort=EFFORT)
    pred = probe.ask(client, probe.prompt(bid, dataset, fut.timestamp.iat[0], identity))
    usage = {f"tokens_{k}": LOCAL.usage[k] for k in ("in", "out", "reasoning", "cached")}
    row = dict(building_id=bid, window_id=wid, dataset=dataset, identity=identity, failed=pred is None,
               llm_calls=LOCAL.usage["llm_calls"], **usage, cost_usd=runner.cost_usd(MODEL, usage))
    metrics = dict.fromkeys(summarize.METRICS)
    if pred is not None:
        metrics.update(probe.forecast_metrics(fut.y_clean.to_numpy(), pred, hist.y_clean.to_numpy()))
    row.update(metrics)
    with LOCK:
        header = not out.exists()
        with out.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if header:
                writer.writeheader()
            writer.writerow(row)


def memorization():
    global PHASE
    PHASE = "memorization"
    status("running")
    out = runner.RESULTS / "memorization_probe_gpt-5.csv"
    done = set()
    if out.exists():
        saved = pd.read_csv(out)
        done = set(zip(saved.building_id, saved.window_id, saved.identity))
    test = pd.read_parquet(runner.WINDOWS["test"])
    groups = list(test.groupby(["building_id", "window_id"]))
    selected = groups[::max(1, len(groups) // 60)][:60]
    cases = [(key, g.sort_values("step"), identity) for key, g in selected
             for identity in (True, False) if (*key, identity) not in done]
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda case: probe_case(case, out), cases))


def main():
    global SPENT, PHASE
    if LEDGER.exists():
        SPENT = sum(json.loads(line)["estimated_usd"] for line in LEDGER.read_text().splitlines())
    LLMClient.chat = metered_chat
    try:
        for variant in VARIANTS:
            forecast("dev", variant, "--limit", "1", "--repeat", "9002")
            scores = runner.RESULTS / f"fc_{MODEL}-{EFFORT}_{variant}_noisy_r9002_dev" / "scores.csv"
            if pd.read_csv(scores).failed.any():
                raise RunStopped(f"Dev smoke check failed for {variant}.")
        for variant in VARIANTS:
            forecast("test", variant, "--subset", "main")
        forecast("test", "weather", "--subset", "main", "--masked-dates")
        for variant in VARIANTS:
            forecast("test", variant, "--subset", "repeat", "--repeat", "1")
        memorization()
        PHASE = "audit and summary"
        status("running")
        for folder in sorted(runner.RESULTS.glob(f"fc_{MODEL}-{EFFORT}_*_test")):
            audit_traces.audit(folder)
        sys.argv = ["summarize", "--split", "test", "--subset", "main"]
        summarize.main()
    except RunStopped as exc:
        status("stopped", str(exc))
        return 2
    except Exception as exc:
        status("failed", type(exc).__name__)
        raise
    status("complete")
    return 0


if __name__ == "__main__":
    with (runner.RESULTS / "gpt5_minimal.lock").open("a+b") as lock:
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            sys.exit("A minimal GPT-5 run is already active.")
        sys.exit(main())
