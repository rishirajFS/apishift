"""Evaluate one model (optionally a LoRA adapter) on Modal; write results + budget log.

    uv run python -m harness.run_baseline --model qwen3-4b --thinking --seeds 1,2 \
        --budget-usd 30 --budget-since 2026-09-25 --planned-usd 1.2

Flow, all inside one ephemeral Modal app (it stops when this process exits,
so nothing idles and bills):
  0. budget guard: logged spend since --budget-since + --planned-usd must fit the cap
  1. download weights on CPU into the Volume (no-op if cached)
  2. start the vLLM server and wait until it is ready
  3. smoke eval (1 episode per mutation type); abort on errors, zero tool calls, or strict tools
  4. full eval for every seed, sharing the one server
Writes results/baseline_<variant>_<split>_s<seed>.json, traces JSONL, and one row
in budget_log.csv per session.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness import budget

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DEFAULT_PROFILE = "teel-lab-ace-ai"
LOGS = REPO / "logs"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


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


def waited_s(notes: list[str]) -> float:
    return next((float(n.split("=")[1]) for n in notes if n.startswith("ready_after_s=")), 0.0)


def decoding(thinking: bool) -> dict:
    """Greedy for non-thinking; Qwen's recommended sampling for thinking mode (greedy loops)."""
    if thinking:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "enable_thinking": True}
    return {"temperature": 0.0, "max_tokens": 1024, "enable_thinking": False}


def run_eval(served_name: str, policy_name: str, base_url: str, split: str, seed: int, max_connections: int,
             limit_per_type: int | None, thinking: bool, log_dir: Path = LOGS, prompt_variant: str = "default",
             sample_time_limit_s: int | None = None):
    from inspect_ai import eval as inspect_eval

    from harness.task import apishift_eval

    dec = decoding(thinking)
    logs = inspect_eval(
        apishift_eval(split=split, seed=seed, limit_per_type=limit_per_type, policy_name=policy_name,
                      prompt_variant=prompt_variant),
        model=f"openai-api/apishift/{served_name}",
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
        log_dir=str(log_dir),
        time_limit=sample_time_limit_s,
        display="plain",
        fail_on_error=False,
        tags=[policy_name, split, f"s{seed}", "smoke" if limit_per_type else "full"],
    )
    return logs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", default="0", help="comma-separated, all evaluated on one server")
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (sampled decoding)")
    parser.add_argument("--lora", default=None, help="NAME=/cache/path adapter to serve and evaluate")
    parser.add_argument("--budget-usd", type=float, default=30.0)
    parser.add_argument("--budget-since", default=None, help="ISO date/time; spend before it is not counted")
    parser.add_argument("--planned-usd", type=float, required=True, help="upper-bound cost of this job")
    parser.add_argument("--queue-timeout-s", type=int, default=10800,
                        help="how long to wait for a GPU to be scheduled; queued time is not billed")
    parser.add_argument("--modal-profile", default=DEFAULT_PROFILE,
                        help="Modal workspace profile; pinned so a changed active profile cannot redirect runs")
    args = parser.parse_args()
    os.environ["MODAL_PROFILE"] = args.modal_profile
    seeds = [int(s) for s in args.seeds.split(",")]

    load_dotenv(REPO / ".env")
    if "APISHIFT_API_KEY" not in os.environ:
        raise SystemExit("APISHIFT_API_KEY missing (.env)")
    budget.guard(args.budget_usd, args.budget_since, args.planned_usd, f"eval {args.model} seeds {seeds}")
    print(f"modal profile: {os.environ['MODAL_PROFILE']}")

    os.environ["APISHIFT_MODEL"] = args.model
    if args.lora:
        os.environ["APISHIFT_LORA"] = args.lora
    import modal

    from harness.results import summarize, write_results
    from infra import modal_vllm as mv

    cfg = mv.MODELS[args.model]
    lora_name = args.lora.split("=", 1)[0] if args.lora else None
    served = lora_name or args.model
    variant = (lora_name or args.model) + ("-thinking" if args.thinking else "")
    t0 = time.time()
    t_ready: float | None = None
    notes: list[str] = [f"seeds={args.seeds}"]
    summaries = {}
    try:
        with modal.enable_output(), mv.app.run():
            print("weights:", mv.download_weights.remote())
            url = mv.serve.get_web_url()
            waited = wait_ready(url, os.environ["APISHIFT_API_KEY"], args.queue_timeout_s)
            t_ready = time.time()
            print(f"server ready after {waited:.0f}s", flush=True)
            notes.append(f"ready_after_s={waited:.0f}")

            smoke_log = run_eval(served, variant, url, args.split, seeds[0], args.max_connections,
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

            for seed in seeds:
                log = run_eval(served, variant, url, args.split, seed, args.max_connections,
                               limit_per_type=None, thinking=args.thinking)
                summary, traces = summarize(log, variant)
                summary |= {"hf_repo": cfg.hf_repo, "lora": args.lora, "gpu": cfg.gpu, "split": args.split,
                            "seed": seed, "decoding": decoding(args.thinking), "vllm": mv.VLLM_VERSION}
                path = write_results(summary, traces, RESULTS, f"baseline_{variant}_{args.split}_s{seed}")
                summaries[seed] = summary
                print("wrote", path, json.dumps({"by_group": {g: v["success_rate"]
                                                              for g, v in summary["by_group"].items()}}))
    finally:
        wall = time.time() - t0
        # GPU is billed from container start, not while queued: count serving time plus at most
        # 15 minutes of startup before readiness. No readiness means no GPU container ran.
        billed = (time.time() - t_ready + min(waited_s(notes), 900)) if t_ready else 0.0
        notes.append(f"wall_s={wall:.0f}")
        est = budget.append(f"eval_{variant}_{args.split}_s{args.seeds.replace(',', '-')}", mv.app.name,
                            cfg.gpu, billed, cfg.usd_per_hour, notes)
        print(f"wall {wall:.0f}s, billed-time estimate {billed:.0f}s, est ${est:.2f}")


if __name__ == "__main__":
    main()
