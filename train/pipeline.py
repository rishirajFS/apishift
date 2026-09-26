"""Debug pipeline in one ART session: self-SFT data -> LoRA SFT -> GRPO, with val checks.

Runs inside a single GPU container (train/modal_art.py). Phases:
  1. val eval of the base model (75 val episodes)
  2. SFT data: sample the base model k times per train episode, keep verified successes
  3. LoRA SFT on those successes, one example per assistant turn (history as the model saw it)
  4. val eval after SFT
  5. GRPO: groups of rollouts on the same (task, mutation, seed), reward = APIShift reward
  6. val eval after GRPO; checkpoint paths and all metrics go to /ckpt/runs/<run_name>/
GRPO stops early when the time budget runs out, so results are always saved
before the Modal timeout.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from apishift.sampling import episodes
from train.art_rollout import Sampling, rollout, sft_turn_examples, trace_line

VAL_GROUPS = ("control", "seen")


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str = "debug-1.7b-v1"
    base_model: str = "Qwen/Qwen3-1.7B"
    seed: int = 0
    sft_samples: int = 4
    sft_keep_per_episode: int = 2
    sft_lr: float = 5e-5
    grpo_steps: int = 20
    groups_per_step: int = 8
    rollouts_per_group: int = 6
    grpo_lr: float = 3e-6  # 1e-5 collapsed a 1.7B run by step ~14 (debug-1.7b-v2)
    val_every: int = 5
    early_stop_ratio: float = 0.5  # stop if val success falls below this fraction of the post-SFT value
    max_tokens: int = 3072
    concurrency: int = 128
    lora_rank: int = 16
    time_budget_s: int = 7200
    art_path: str = "/ckpt/art"
    out_root: str = "/ckpt/runs"
    # smoke: exercise every phase on a handful of episodes (cheap GPU check before a real run)
    smoke: bool = False
    # skip_sft: GRPO straight from the base model (self-SFT did not help on 1.7B, v2 and v3)
    skip_sft: bool = False


class Run:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.t0 = time.monotonic()
        self.out = Path(cfg.out_root) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.metrics: list[dict[str, Any]] = []
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.sampling = Sampling(max_tokens=cfg.max_tokens)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def log(self, phase: str, **values: Any) -> None:
        row = {"phase": phase, "elapsed_s": round(self.elapsed()), **values}
        self.metrics.append(row)
        print("METRIC", json.dumps(row), flush=True)
        (self.out / "metrics.json").write_text(json.dumps(self.metrics, indent=1))

    def save_traces(self, name: str, traces: list[dict]) -> None:
        with (self.out / f"traces_{name}.jsonl").open("w") as fh:
            for t in traces:
                fh.write(trace_line(t) + "\n")

    async def one(self, model, task, spec, name: str):
        async with self.sem:
            return await rollout(model, task, spec, self.cfg.seed, self.sampling, name)


def group_of(mtype: str) -> str:
    return "control" if mtype == "none" else "seen"


def summarize(traces: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(traces)}
    if not traces:
        return out
    out["success"] = round(sum(t["success"] for t in traces) / len(traces), 4)
    out["reward"] = round(sum(t["reward"]["total"] for t in traces) / len(traces), 4)
    out["turns"] = round(sum(t["turns"] for t in traces) / len(traces), 3)
    for g in VAL_GROUPS:
        sub = [t for t in traces if group_of(t["mutation"]["type"]) == g]
        if sub:
            out[f"success_{g}"] = round(sum(t["success"] for t in sub) / len(sub), 4)
    return out


async def val_eval(run: Run, model, label: str) -> dict[str, Any]:
    pairs = episodes("val", run.cfg.seed)[: 3 if run.cfg.smoke else None]
    results = await asyncio.gather(*(run.one(model, t, s, label) for t, s in pairs), return_exceptions=True)
    traces = [r[1] for r in results if not isinstance(r, BaseException)]
    errors = sum(isinstance(r, BaseException) for r in results)
    run.save_traces(f"val_{label}", traces)
    stats = summarize(traces) | {"errors": errors, "step": await model.get_step()}
    run.log(f"val_{label}", **stats)
    return stats


async def sft_phase(run: Run, model) -> dict[str, Any]:
    import art

    cfg = run.cfg
    pairs = episodes("train", cfg.seed)[: 6 if cfg.smoke else None]
    jobs = [run.one(model, t, s, "sft-sample") for t, s in pairs for _ in range(cfg.sft_samples)]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    traces = [r[1] for r in results if not isinstance(r, BaseException)]
    run.save_traces("sft_samples", traces)

    kept: dict[str, int] = {}
    examples = []
    for t in traces:
        key = t["episode_id"]
        if t["success"] and t["reward"]["total"] >= 0.9 and kept.get(key, 0) < cfg.sft_keep_per_episode:
            kept[key] = kept.get(key, 0) + 1
            examples += [art.Trajectory(messages_and_choices=ex, tools=t["tools"], reward=1.0)
                         for ex in sft_turn_examples(t)]
    stats = {"samples": len(traces), "errors": len(results) - len(traces),
             "sample_success": round(sum(t["success"] for t in traces) / max(len(traces), 1), 4),
             "episodes_with_success": len(kept), "episodes_total": len(pairs), "sft_examples": len(examples)}
    run.log("sft_data", **stats)
    if not examples and cfg.smoke:
        # smoke only checks the SFT path runs: fall back to an oracle-quality example
        examples = [art.Trajectory(messages_and_choices=ex, tools=traces[0]["tools"], reward=1.0)
                    for ex in sft_turn_examples(traces[0])[:1]]
    if not examples:
        raise RuntimeError("no successful trajectories to fine-tune on")

    random.Random(cfg.seed).shuffle(examples)
    await model.train_sft(examples, config=art.TrainSFTConfig(learning_rate=cfg.sft_lr, assistant_turns="last"),
                          log_metrics=False)
    run.log("sft_done", step=await model.get_step())
    return stats


async def grpo_phase(run: Run, model, backend, sft_val: float) -> dict[str, Any]:
    """GRPO with a val check every `val_every` steps; returns the best step by val success."""
    import art

    cfg = run.cfg
    history: list[dict[str, Any]] = []
    pairs = episodes("train", cfg.seed)
    rng = random.Random(cfg.seed + 1)
    # budget check before each step: stop while there is time left for the final val eval
    last_step_s = 0.0
    for step in range(cfg.grpo_steps):
        if run.elapsed() + 2 * last_step_s + 900 > cfg.time_budget_s:
            run.log("grpo_stopped_for_time", step=step)
            break
        started = time.monotonic()
        batch = rng.sample(pairs, cfg.groups_per_step)
        traces: list[dict] = []

        async def traj(task, spec, sink=traces):
            tr, trace = await run.one(model, task, spec, "grpo")
            sink.append(trace)
            return tr

        groups = await art.gather_trajectory_groups(
            [art.TrajectoryGroup(traj(t, s) for _ in range(cfg.rollouts_per_group)) for t, s in batch],
            max_exceptions=cfg.rollouts_per_group * cfg.groups_per_step,
        )
        rewards = [[tr.reward for tr in g.trajectories] for g in groups]
        informative = sum(1 for r in rewards if r and max(r) > min(r))
        result = await backend.train(model, groups, learning_rate=cfg.grpo_lr)
        last_step_s = time.monotonic() - started
        run.log("grpo_step", step=result.step, step_s=round(last_step_s),
                reward=round(sum(map(sum, rewards)) / max(sum(map(len, rewards)), 1), 4),
                success=round(sum(t["success"] for t in traces) / max(len(traces), 1), 4),
                informative_groups=informative, groups=len(groups),
                train={k: v for k, v in list(result.metrics.items())[:12] if isinstance(v, (int, float))})
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.grpo_steps:
            val = await val_eval(run, model, f"grpo_s{result.step}")
            history.append({"step": result.step, "success": val.get("success", 0.0)})
            if val.get("success", 0.0) < cfg.early_stop_ratio * sft_val:
                run.log("grpo_early_stop", step=result.step, val_success=val.get("success"), sft_val=sft_val)
                break
    best = max(history, key=lambda h: h["success"], default=None)
    return {"history": history, "best": best}


def checkpoint_dirs(cfg: PipelineConfig) -> list[str]:
    run_dir = Path(cfg.art_path) / "apishift" / "models" / cfg.run_name / "checkpoints"
    return sorted(str(p) for p in run_dir.glob("*") if p.is_dir())


async def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    import art
    from art.local import LocalBackend

    run = Run(cfg)
    (run.out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    backend = LocalBackend(path=cfg.art_path)
    model = art.TrainableModel(
        name=cfg.run_name, run_name=cfg.run_name, project="apishift", base_model=cfg.base_model,
        base_path=cfg.art_path,
        lora_config={"rank": cfg.lora_rank, "alpha": 2 * cfg.lora_rank},
        _internal_config={"chat_template_kwargs": {"enable_thinking": True},
                          "init_args": {"max_seq_length": 16384},
                          "engine_args": {"max_model_len": 16384, "gpu_memory_utilization": 0.75}},
    )
    try:
        await model.register(backend)
        run.log("registered", step=await model.get_step())
        base = await val_eval(run, model, "base")
        if cfg.skip_sft:
            sft_stats, sft_step, sft = {"skipped": True}, await model.get_step(), base
        else:
            sft_stats = await sft_phase(run, model)
            sft_step = await model.get_step()
            sft = await val_eval(run, model, "sft")
        grpo_info = await grpo_phase(run, model, backend, sft.get("success", 0.0))
        grpo = await val_eval(run, model, "grpo")
        summary = {"config": asdict(cfg), "val": {"base": base, "sft": sft, "grpo": grpo}, "sft_data": sft_stats,
                   "sft_step": sft_step, "final_step": await model.get_step(), "grpo_val": grpo_info,
                   "best_step": (grpo_info["best"] or {}).get("step"),
                   "model_dir": str(Path(cfg.art_path) / "apishift" / "models" / cfg.run_name),
                   "checkpoints": checkpoint_dirs(cfg),
                   "elapsed_s": round(run.elapsed()), "metrics": run.metrics}
        (run.out / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary
    finally:
        await backend.close()
