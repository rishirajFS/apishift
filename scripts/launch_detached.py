"""Launch Modal jobs that keep running without this terminal or an internet connection.

    uv run python scripts/launch_detached.py --pilot --prompt-baseline \
        --budget-usd 11 --budget-since 2026-09-26

Each job runs in a detached Modal app: after the spawn returns, this process can
exit and the laptop can sleep. Results, logs and job records go to the
`apishift-checkpoints` Volume (runs/, evals/, jobs/). Pull them later with
scripts/collect_detached.py. The budget guard uses worst-case costs: GPU hard
caps (Modal timeouts / eval time budget) plus the CPU orchestrators.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import budget  # noqa: E402

CPU_USD_PER_HOUR = 0.063  # 1 core + 2 GiB on Modal
PILOT_TIMEOUT_S = 6900
EVAL_QUEUE_S, EVAL_MAX_S, EVAL_SAMPLE_LIMIT_S = 3 * 3600, 3600, 900
EVAL_JOB_TIMEOUT_S = EVAL_QUEUE_S + 2 * 3600


def pilot_plan(run_name: str) -> tuple[dict, float, float]:
    cfg = {"run_name": run_name, "base_model": "Qwen/Qwen3-4B", "skip_sft": True, "grpo_steps": 20,
           "groups_per_step": 8, "rollouts_per_group": 6, "grpo_lr": 3e-6, "val_every": 5,
           "time_budget_s": PILOT_TIMEOUT_S - 600}
    gpu = PILOT_TIMEOUT_S / 3600 * 3.95 * 1.1
    cpu = (PILOT_TIMEOUT_S + 3 * 3600 + 1800) / 3600 * CPU_USD_PER_HOUR
    return cfg, round(gpu, 2), round(cpu, 2)


REPILOT_TIMEOUT_S = 8100


def repilot_plan(run_name: str) -> tuple[dict, float, float]:
    """4B GRPO from base on the adaptation-hard pool (scan 4 samples/episode, then 30 steps)."""
    cfg = {"run_name": run_name, "base_model": "Qwen/Qwen3-4B", "skip_sft": True, "pool_scan": True,
           "scan_samples": 4, "grpo_steps": 30, "groups_per_step": 8, "rollouts_per_group": 6,
           "grpo_lr": 3e-6, "val_every": 10, "time_budget_s": REPILOT_TIMEOUT_S - 600}
    gpu = REPILOT_TIMEOUT_S / 3600 * 3.95 * 1.1
    cpu = (REPILOT_TIMEOUT_S + 3 * 3600 + 1800) / 3600 * CPU_USD_PER_HOUR
    return cfg, round(gpu, 2), round(cpu, 2)


def checkpoint_eval_plan(run_name: str, lora_name: str) -> tuple[dict, float, float]:
    """Test-split eval (3 seeds, default prompt) of a finished run's best checkpoint, served as LoRA."""
    summary = json.loads((REPO / "results" / "remote" / run_name / run_name / "summary.json").read_text())
    step = summary.get("best_step") or summary["final_step"]
    by_step = {int(Path(c).name): c for c in summary["checkpoints"] if Path(c).name.isdigit()}
    lora = f"{lora_name}={by_step[step]}"
    job = {"name": f"eval-{lora_name}", "model": "qwen3-4b", "served": lora_name, "lora": lora,
           "variant": f"{lora_name}-thinking", "split": "test", "seeds": [0, 1, 2], "thinking": True,
           "prompt_variant": "default", "max_connections": 48, "queue_timeout_s": EVAL_QUEUE_S,
           "max_eval_s": EVAL_MAX_S, "sample_time_limit_s": EVAL_SAMPLE_LIMIT_S, "source_step": step}
    gpu = (EVAL_MAX_S + 1800 + 900) / 3600 * 0.80 * 1.1
    cpu = EVAL_JOB_TIMEOUT_S / 3600 * CPU_USD_PER_HOUR
    return job, round(gpu, 2), round(cpu, 2)


def prompt_baseline_plan(name: str) -> tuple[dict, float, float]:
    job = {"name": name, "model": "qwen3-4b", "served": "qwen3-4b", "variant": "qwen3-4b-thinking-recovery",
           "split": "test", "seeds": [0, 1, 2], "thinking": True, "prompt_variant": "recovery",
           "max_connections": 48, "queue_timeout_s": EVAL_QUEUE_S, "max_eval_s": EVAL_MAX_S,
           "sample_time_limit_s": EVAL_SAMPLE_LIMIT_S}
    # worst case: eval budget + one more seed (~30 min) + 15 min startup, on an L4
    gpu = (EVAL_MAX_S + 1800 + 900) / 3600 * 0.80 * 1.1
    cpu = EVAL_JOB_TIMEOUT_S / 3600 * CPU_USD_PER_HOUR
    return job, round(gpu, 2), round(cpu, 2)


def log_row(job: str, app: str, gpu: str, planned: float, notes: str) -> None:
    import csv

    new = not budget.BUDGET_LOG.exists()
    with budget.BUDGET_LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=budget.COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"date": dt.datetime.now().isoformat(timespec="seconds"), "job": job, "modal_app": app,
                    "gpu": gpu, "duration_s": "", "est_cost_usd": planned, "actual_cost_usd": "", "notes": notes})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", action="store_true", help="4B GRPO pilot from base, H100")
    p.add_argument("--prompt-baseline", action="store_true", help="4B thinking + recovery prompt, 3 seeds, L4")
    p.add_argument("--repilot", action="store_true", help="4B GRPO on the adaptation-hard pool, H100")
    p.add_argument("--repilot-name", default="grpo-4b-pool-v1")
    p.add_argument("--checkpoint-eval", default=None, metavar="RUN_NAME",
                   help="eval the best checkpoint of a finished run (needs results/remote/<run>/ collected)")
    p.add_argument("--pilot-name", default="grpo-4b-pilot-v1")
    p.add_argument("--baseline-name", default="prompt-recovery-4b-v1")
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--budget-since", required=True)
    p.add_argument("--modal-profile", default="teel-lab-ace-ai")
    p.add_argument("--launches-file", type=Path, default=None, help="default: results/launches/<today>.json")
    args = p.parse_args()

    plans = []
    if args.pilot:
        plans.append(("pilot", *pilot_plan(args.pilot_name)))
    if args.prompt_baseline:
        plans.append(("prompt_baseline", *prompt_baseline_plan(args.baseline_name)))
    if args.repilot:
        plans.append(("pilot", *repilot_plan(args.repilot_name)))
    if args.checkpoint_eval:
        plans.append(("checkpoint_eval", *checkpoint_eval_plan(args.checkpoint_eval, "grpo-4b")))
    if not plans:
        raise SystemExit("nothing to launch")
    planned = round(sum(g + c for _, _, g, c in plans), 2)
    budget.guard(args.budget_usd, args.budget_since, planned, f"detached launch {[n for n, *_ in plans]}")

    os.environ["MODAL_PROFILE"] = args.modal_profile
    os.environ["APISHIFT_TRAIN_GPU"] = "H100"
    os.environ["APISHIFT_TRAIN_TIMEOUT_S"] = str(REPILOT_TIMEOUT_S if args.repilot else PILOT_TIMEOUT_S)
    os.environ["APISHIFT_MODEL"] = "qwen3-4b"
    os.environ["APISHIFT_EVAL_JOB_TIMEOUT_S"] = str(EVAL_JOB_TIMEOUT_S)
    import modal

    launched = {}
    for name, spec, gpu_usd, cpu_usd in plans:
        if name == "pilot":
            from train import modal_art as mod

            with modal.enable_output(), mod.app.run(detach=True):
                call = mod.train_job.spawn(spec)
            app_name, gpu = mod.app.name, "H100"
        else:
            if spec.get("lora"):
                os.environ["APISHIFT_LORA"] = spec["lora"]
            from infra import modal_vllm as mod

            with modal.enable_output(), mod.app.run(detach=True):
                call = mod.eval_job.spawn(spec)
            app_name, gpu = mod.app.name, "L4"
        launched[name] = {"call_id": call.object_id, "app": app_name, "spec": spec,
                          "planned_gpu_usd": gpu_usd, "planned_cpu_usd": cpu_usd}
        log_row(f"detached_{name}_{spec.get('run_name') or spec.get('name')}", app_name, gpu,
                round(gpu_usd + cpu_usd, 2), f"detached launch; call {call.object_id}; worst-case estimate, "
                "actual from billing later")
        print(f"launched {name}: call {call.object_id} on app {app_name}", flush=True)

    out = args.launches_file or REPO / "results" / "launches" / f"{dt.date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**existing, **launched}, indent=1) + "\n")
    print("recorded", out, f"(modal profile {args.modal_profile})")


if __name__ == "__main__":
    main()
