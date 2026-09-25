"""Run the ART debug pipeline on Modal with a hard spend cap.

    uv run python -m train.run_debug --run-name debug-1.7b-v1 --timeout-s 9000 \
        --budget-usd 30 --budget-since 2026-09-25

The GPU function's timeout is the cap: planned cost = timeout x GPU rate x 1.1,
and the budget guard refuses to start if that does not fit. The pipeline
stops GRPO early to leave time for its final val eval before the timeout.
Writes results/train/<run>.json and a budget_log.csv row.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from harness import budget

REPO = Path(__file__).resolve().parents[1]


def prefetch_only(args, modal, modal_art) -> None:
    budget.guard(args.budget_usd, args.budget_since, 0.25, "train image build + weight prefetch (CPU)")
    t0 = time.time()
    try:
        with modal.enable_output(), modal_art.app.run():
            print("prefetched:", modal_art.prefetch.remote(args.base_model), flush=True)
    finally:
        est = budget.append(f"prefetch_{args.run_name}", modal_art.app.name, "CPU", time.time() - t0, 0.20,
                            ["image build + weights, no GPU"])
        print(f"wall {time.time() - t0:.0f}s, est ${est:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--gpu", default="H100")
    p.add_argument("--timeout-s", type=int, default=9000)
    p.add_argument("--grpo-steps", type=int, default=20)
    p.add_argument("--groups-per-step", type=int, default=8)
    p.add_argument("--rollouts-per-group", type=int, default=6)
    p.add_argument("--sft-samples", type=int, default=4)
    p.add_argument("--grpo-lr", type=float, default=3e-6)
    p.add_argument("--budget-usd", type=float, default=30.0)
    p.add_argument("--budget-since", default=None)
    p.add_argument("--modal-profile", default="teel-lab-ace-ai")
    p.add_argument("--prefetch-only", action="store_true", help="build the image and cache weights on CPU only")
    p.add_argument("--smoke", action="store_true", help="every phase on a few episodes (cheap GPU check)")
    args = p.parse_args()

    os.environ["MODAL_PROFILE"] = args.modal_profile
    os.environ["APISHIFT_TRAIN_GPU"] = args.gpu
    os.environ["APISHIFT_TRAIN_TIMEOUT_S"] = str(args.timeout_s)
    import modal

    from train import modal_art

    rate = modal_art.GPU_USD_PER_HOUR
    planned = round(args.timeout_s / 3600 * rate * 1.1, 2)
    if args.prefetch_only:
        return prefetch_only(args, modal, modal_art)
    what = f"train {args.run_name} ({args.gpu}, {args.timeout_s}s cap)"
    budget.guard(args.budget_usd, args.budget_since, planned, what)
    print(f"modal profile: {os.environ['MODAL_PROFILE']}")

    cfg = {"run_name": args.run_name, "base_model": args.base_model, "grpo_steps": args.grpo_steps,
           "groups_per_step": args.groups_per_step, "rollouts_per_group": args.rollouts_per_group,
           "sft_samples": args.sft_samples, "grpo_lr": args.grpo_lr,
           "time_budget_s": args.timeout_s - 600, "smoke": args.smoke}
    t0 = time.time()
    notes = [f"timeout_s={args.timeout_s}"]
    try:
        with modal.enable_output(), modal_art.app.run():
            print("prefetched:", modal_art.prefetch.remote(args.base_model), flush=True)
            summary = modal_art.train.remote(cfg)
        out = REPO / "results" / "train" / f"{args.run_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1) + "\n")
        print("wrote", out)
        print(json.dumps(summary["val"], indent=1))
        print("checkpoints:", summary["checkpoints"][-3:])
    except Exception as exc:
        notes.append(f"error={type(exc).__name__}")
        raise
    finally:
        est = budget.append(f"train_{args.run_name}", modal_art.app.name, args.gpu, time.time() - t0, rate, notes)
        print(f"wall {time.time() - t0:.0f}s, est ${est:.2f}")


if __name__ == "__main__":
    main()
