"""Run a baseline eval for one model on Modal and write results + budget log.

    uv run python -m harness.run_baseline --model qwen3-1.7b

Flow, all inside one ephemeral Modal app (it stops when this process exits,
so nothing idles and bills):
  1. download weights on CPU into the Volume (no-op if cached)
  2. start the vLLM server and wait until it is ready
  3. smoke eval (1 episode per mutation type); abort on any error
  4. full eval on the split
Writes results/baseline_<model>_<split>_s<seed>.json, traces JSONL, and a row
in budget_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUDGET_LOG = REPO / "budget_log.csv"
RESULTS = REPO / "results"
LOGS = REPO / "logs"
BUDGET_COLUMNS = ["date", "job", "modal_app", "gpu", "duration_s", "est_cost_usd", "actual_cost_usd", "notes"]
SPEND_STOP_USD = 15.0


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def logged_spend() -> float:
    if not BUDGET_LOG.exists():
        return 0.0
    with BUDGET_LOG.open() as fh:
        rows = list(csv.DictReader(fh))
    return sum(float(r["actual_cost_usd"] or r["est_cost_usd"] or 0) for r in rows)


def append_budget(row: dict) -> None:
    new = not BUDGET_LOG.exists()
    with BUDGET_LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BUDGET_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def wait_ready(base_url: str, api_key: str, timeout_s: int) -> float:
    """Poll /v1/models until the server answers. Returns seconds waited."""
    start = time.time()
    req = urllib.request.Request(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 200:
                    return time.time() - start
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(10)
    raise TimeoutError(f"server not ready after {timeout_s}s")


def strict_tools_sent(log) -> list[str]:
    """Names of tools sent with strict=true in any model request of an eval log."""
    found = set()
    for sample in log.samples or []:
        for ev in sample.events:
            call = getattr(ev, "call", None) if ev.event == "model" else None
            for tool in (call.request.get("tools", []) if call else []):
                if tool.get("function", {}).get("strict"):
                    found.add(tool["function"]["name"])
    return sorted(found)


def decoding(thinking: bool) -> dict:
    """Greedy for non-thinking; Qwen's recommended sampling for thinking mode (greedy loops)."""
    if thinking:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "enable_thinking": True}
    return {"temperature": 0.0, "max_tokens": 1024, "enable_thinking": False}


def run_eval(model_key: str, base_url: str, split: str, seed: int, max_connections: int,
             limit_per_type: int | None, thinking: bool):
    from inspect_ai import eval as inspect_eval

    from harness.task import apishift_eval

    dec = decoding(thinking)
    policy_name = f"{model_key}-thinking" if thinking else model_key
    logs = inspect_eval(
        apishift_eval(split=split, seed=seed, limit_per_type=limit_per_type, policy_name=policy_name),
        model=f"openai-api/apishift/{model_key}",
        model_base_url=f"{base_url}/v1",
        # Inspect defaults to strict tools; vLLM then grammar-constrains arguments to the STALE
        # schema, which makes renamed params, new fields and new endpoints impossible to emit.
        model_args={"strict_tools": False},
        max_connections=max_connections,
        temperature=dec["temperature"],
        top_p=dec.get("top_p"),
        top_k=dec.get("top_k"),
        max_tokens=dec["max_tokens"],
        seed=seed,
        extra_body={"chat_template_kwargs": {"enable_thinking": dec["enable_thinking"]}},
        log_dir=str(LOGS),
        display="plain",
        fail_on_error=False,
        tags=[policy_name, split, "smoke" if limit_per_type else "full"],
    )
    return logs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (sampled decoding)")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    if "APISHIFT_API_KEY" not in os.environ:
        raise SystemExit("APISHIFT_API_KEY missing (.env)")
    spent = logged_spend()
    if spent >= SPEND_STOP_USD:
        raise SystemExit(f"logged spend ${spent:.2f} >= ${SPEND_STOP_USD}; ask before spending more")

    os.environ["APISHIFT_MODEL"] = args.model
    import modal

    from harness.results import summarize, write_results
    from infra import modal_vllm as mv

    cfg = mv.MODELS[args.model]
    variant = f"{args.model}-thinking" if args.thinking else args.model
    run_name = f"baseline_{variant}_{args.split}_s{args.seed}"
    t0 = time.time()
    notes = []
    summary = None
    try:
        with modal.enable_output(), mv.app.run():
            print("weights:", mv.download_weights.remote())
            url = mv.serve.get_web_url()
            waited = wait_ready(url, os.environ["APISHIFT_API_KEY"], mv.STARTUP_TIMEOUT_S)
            print(f"server ready after {waited:.0f}s")
            notes.append(f"ready_after_s={waited:.0f}")

            smoke_log = run_eval(args.model, url, args.split, args.seed, args.max_connections,
                                 limit_per_type=1, thinking=args.thinking)
            smoke, smoke_traces = summarize(smoke_log, variant)
            n_calls = sum(len(t["calls"]) for t in smoke_traces)
            print("smoke:", json.dumps({"n_errors": smoke["n_errors"], "tool_calls": n_calls,
                                        "by_group": smoke["by_group"]}))
            strict = strict_tools_sent(smoke_log)
            if strict:
                notes.append("aborted_strict_tools")
                raise SystemExit(f"tools were sent with strict=true: {strict[:5]}")
            if smoke["n_errors"] or n_calls == 0:
                notes.append("aborted_after_smoke")
                raise SystemExit(f"smoke failed: errors={smoke['errors']} tool_calls={n_calls}")
            if args.smoke_only:
                return

            log = run_eval(args.model, url, args.split, args.seed, args.max_connections,
                           limit_per_type=None, thinking=args.thinking)
            summary, traces = summarize(log, variant)
            summary |= {"hf_repo": cfg.hf_repo, "gpu": cfg.gpu, "split": args.split, "seed": args.seed,
                        "decoding": decoding(args.thinking), "vllm": mv.VLLM_VERSION}
            path = write_results(summary, traces, RESULTS, run_name)
            print("wrote", path)
    finally:
        duration = time.time() - t0
        est = round(duration / 3600 * cfg.usd_per_hour * 1.1, 2)  # +10% for CPU/memory
        append_budget({
            "date": dt.datetime.now().isoformat(timespec="seconds"),
            "job": run_name, "modal_app": mv.app.name, "gpu": cfg.gpu,
            "duration_s": round(duration), "est_cost_usd": est, "actual_cost_usd": "",
            "notes": ";".join(notes),
        })
        print(f"wall {duration:.0f}s, est ${est:.2f} (upper bound: GPU billed from server start)")
    if summary:
        print(json.dumps({"overall": summary["overall"], "by_group": summary["by_group"]}, indent=1))


if __name__ == "__main__":
    main()
