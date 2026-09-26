"""Eval job that runs entirely on Modal (no local terminal or network needed after launch).

Called inside infra/modal_vllm.eval_job: waits for the vLLM web server in the
same app, runs the smoke gate and then every seed, and writes results, traces
and Inspect logs to the checkpoints Volume under evals/<name>/. A job.json
record carries timestamps and a GPU-time estimate for budget reconciliation.
GPU spend is bounded by `max_eval_s` (no new seed starts after it) plus a
per-episode time limit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from harness.results import summarize, write_results
from harness.run_baseline import decoding, run_eval, strict_tools_sent, wait_ready


def _record(path: Path, **values: Any) -> None:
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **values}, indent=1))


def run_job(job: dict[str, Any], url: str, out_root: str, usd_per_hour: float, commit) -> dict[str, Any]:
    os.environ.setdefault("APISHIFT_API_KEY", os.environ["VLLM_API_KEY"])
    out = Path(out_root) / "evals" / job["name"]
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rec = out / "job.json"
    _record(rec, job=job, launched=time.time(), status="waiting_for_gpu")
    commit()

    t_ready = None
    waited = 0.0
    summaries: dict[int, Any] = {}
    status = "failed"
    try:
        waited = wait_ready(url, os.environ["APISHIFT_API_KEY"], job["queue_timeout_s"])
        t_ready = time.time()
        _record(rec, ready_after_s=round(waited), status="running")
        commit()

        common = dict(served_name=job["served"], policy_name=job["variant"], base_url=url, split=job["split"],
                      max_connections=job["max_connections"], thinking=job["thinking"], log_dir=out / "logs",
                      prompt_variant=job["prompt_variant"], sample_time_limit_s=job["sample_time_limit_s"])
        smoke_log = run_eval(seed=job["seeds"][0], limit_per_type=1, **common)
        smoke, smoke_traces = summarize(smoke_log, job["variant"])
        n_calls = sum(len(t["calls"]) for t in smoke_traces)
        strict = strict_tools_sent(smoke_log)
        _record(rec, smoke={"n_errors": smoke["n_errors"], "tool_calls": n_calls, "strict_tools": strict})
        if strict or smoke["n_errors"] or n_calls == 0:
            raise RuntimeError(f"smoke gate failed: strict={strict} errors={smoke['errors']} calls={n_calls}")

        for seed in job["seeds"]:
            if time.time() - t_ready > job["max_eval_s"]:
                _record(rec, stopped_for_time_before_seed=seed)
                break
            log = run_eval(seed=seed, limit_per_type=None, **common)
            summary, traces = summarize(log, job["variant"])
            summary |= {"model": job["model"], "lora": job.get("lora"), "split": job["split"], "seed": seed,
                        "prompt_variant": job["prompt_variant"], "decoding": decoding(job["thinking"])}
            write_results(summary, traces, out / "results", f"baseline_{job['variant']}_{job['split']}_s{seed}")
            summaries[seed] = {g: v["success_rate"] for g, v in summary["by_group"].items()}
            _record(rec, by_seed=summaries)
            commit()
        status = "done"
        return {"name": job["name"], "by_seed": summaries}
    finally:
        ended = time.time()
        gpu_s = (ended - t_ready + min(waited, 900)) if t_ready else 0.0
        _record(rec, ended=ended, status=status, gpu_seconds_est=round(gpu_s),
                est_gpu_usd=round(gpu_s / 3600 * usd_per_hour, 2))
        commit()
