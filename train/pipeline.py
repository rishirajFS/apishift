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
from train.state import RunState, config_hash

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
    # pool_scan: before GRPO, sample each train episode `scan_samples` times with the starting
    # policy and train only on adaptation-hard episodes (train/pool.py). The 4B pilot (unfiltered)
    # had too few groups with reward variance because base 4B already solves most train episodes.
    pool_scan: bool = False
    scan_samples: int = 4
    pool_lo: float = 0.1
    pool_hi: float = 0.9
    control_min: float = 0.5
    min_pool: int = 24
    # GPU seconds allowed across ALL attempts (Modal restarts preempted GPU functions); 0 = time_budget_s
    total_gpu_budget_s: int = 0
    # system prompt for every rollout and val episode ("recovery" = the prompting baseline's prompt)
    prompt_variant: str = "default"


class Run:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.t0 = time.monotonic()
        self.out = Path(cfg.out_root) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        prev = self.out / "metrics.json"
        self.metrics: list[dict[str, Any]] = json.loads(prev.read_text()) if prev.exists() else []
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
            return await rollout(model, task, spec, self.cfg.seed, self.sampling, name, self.cfg.prompt_variant)


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


def load_pool(run: Run) -> list[tuple[Any, Any]]:
    keep = {(p["task_id"], p["mutation_type"]) for p in json.loads((run.out / "pool.json").read_text())}
    return [(t, s) for t, s in episodes("train", run.cfg.seed) if (t.id, s.type) in keep]


async def scan_phase(run: Run, model) -> list[tuple[Any, Any]]:
    """Sample every train episode k times; return the adaptation-hard (task, spec) pool."""
    from train.pool import select_pool

    cfg = run.cfg
    pairs = episodes("train", cfg.seed)
    jobs = [run.one(model, t, s, "scan") for t, s in pairs for _ in range(cfg.scan_samples)]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    traces = [r[1] for r in results if not isinstance(r, BaseException)]
    run.save_traces("scan", traces)
    pool = select_pool(traces, lo=cfg.pool_lo, hi=cfg.pool_hi, control_min=cfg.control_min,
                       min_size=cfg.min_pool)
    (run.out / "pool.json").write_text(json.dumps(pool, indent=1))
    by_type: dict[str, int] = {}
    for p in pool:
        by_type[p["mutation_type"]] = by_type.get(p["mutation_type"], 0) + 1
    run.log("pool", size=len(pool), by_type=by_type, scanned=len(traces), errors=len(results) - len(traces),
            scan_success=round(sum(t["success"] for t in traces) / max(len(traces), 1), 4),
            mean_pool_rate=round(sum(p["success_rate"] for p in pool) / len(pool), 4))
    keep = {(p["task_id"], p["mutation_type"]) for p in pool}
    return [(t, s) for t, s in pairs if (t.id, s.type) in keep]


async def grpo_phase(run: Run, model, backend, sft_val: float, pool: list[tuple[Any, Any]] | None,
                     state: RunState, persist) -> dict[str, Any]:
    """GRPO with a val check every `val_every` steps; resumes after a restart from ART's step.

    Returns the val history and the best step by val reward.
    """
    import art

    cfg = run.cfg
    pairs = pool if pool else episodes("train", cfg.seed)
    state.begin_grpo(await model.get_step())
    persist()
    remaining = state.remaining_steps(cfg.grpo_steps, await model.get_step())
    budget_s = cfg.total_gpu_budget_s or cfg.time_budget_s
    run.log("grpo_resume", start_step=state.grpo_start_step, remaining=remaining, attempt=state.attempt)
    last_step_s = 0.0
    for i in range(remaining):
        done = cfg.grpo_steps - remaining + i  # GRPO steps finished before this one, across attempts
        state.record_gpu(run.elapsed())
        if (run.elapsed() + 2 * last_step_s + 900 > cfg.time_budget_s
                or state.gpu_seconds_total() + 2 * last_step_s + 900 > budget_s):
            run.log("grpo_stopped_for_time", step=done, gpu_s_total=round(state.gpu_seconds_total()))
            break
        started = time.monotonic()
        rng = random.Random(cfg.seed * 100_003 + done)  # per-step seed: a resumed run samples the same batches
        batch = (rng.sample(pairs, cfg.groups_per_step) if len(pairs) >= cfg.groups_per_step
                 else rng.choices(pairs, k=cfg.groups_per_step))
        traces: list[dict] = []

        async def traj(task, spec, sink=traces):
            tr, trace = await run.one(model, task, spec, "grpo")
            sink.append(trace)
            return tr

        groups = await art.gather_trajectory_groups(
            [art.TrajectoryGroup(traj(t, s) for _ in range(cfg.rollouts_per_group)) for t, s in batch],
            max_exceptions=cfg.rollouts_per_group * cfg.groups_per_step,
        )
        with (run.out / "traces_grpo.jsonl").open("a") as fh:
            for t in traces:
                fh.write(trace_line({**t, "grpo_step": done, "attempt": state.attempt}) + "\n")
        rewards = [[tr.reward for tr in g.trajectories] for g in groups]
        informative = sum(1 for r in rewards if r and max(r) > min(r))
        result = await backend.train(model, groups, learning_rate=cfg.grpo_lr)
        last_step_s = time.monotonic() - started
        run.log("grpo_step", step=result.step, grpo_step=done + 1, step_s=round(last_step_s),
                reward=round(sum(map(sum, rewards)) / max(sum(map(len, rewards)), 1), 4),
                success=round(sum(t["success"] for t in traces) / max(len(traces), 1), 4),
                informative_groups=informative, groups=len(groups),
                train={k: v for k, v in list(result.metrics.items())[:12] if isinstance(v, (int, float))})
        stop = False
        if (done + 1) % cfg.val_every == 0 or done + 1 == cfg.grpo_steps:
            val = await val_eval(run, model, f"grpo_s{result.step}")
            state.history.append({"step": result.step, "success": val.get("success", 0.0),
                                  "reward": val.get("reward", 0.0)})
            if val.get("success", 0.0) < cfg.early_stop_ratio * sft_val:
                run.log("grpo_early_stop", step=result.step, val_success=val.get("success"), sft_val=sft_val)
                stop = True
        state.record_gpu(run.elapsed())
        persist()
        if stop:
            break
    # reward, not success: val success sits near the ceiling for 4B (88% at base)
    best = max(state.history, key=lambda h: (h["reward"], h["success"]), default=None)
    return {"history": state.history, "best": best}


def checkpoint_dirs(cfg: PipelineConfig) -> list[str]:
    run_dir = Path(cfg.art_path) / "apishift" / "models" / cfg.run_name / "checkpoints"
    return sorted(str(p) for p in run_dir.glob("*") if p.is_dir())


RESUME_IGNORED_FIELDS = ("time_budget_s", "total_gpu_budget_s", "concurrency")


async def run_pipeline(cfg: PipelineConfig, commit=lambda: None) -> dict[str, Any]:
    """Run (or resume after a preemption restart) the pipeline. `commit` persists the Volume."""
    import art
    from art.local import LocalBackend

    run = Run(cfg)
    (run.out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    state = RunState.load(run.out / "state.json",
                          config_hash({k: v for k, v in asdict(cfg).items() if k not in RESUME_IGNORED_FIELDS}))
    if state.attempt > 1 and not cfg.skip_sft:
        raise RuntimeError("resume after restart is only supported with skip_sft=True")

    def persist() -> None:
        state.record_gpu(run.elapsed())
        state.save()
        commit()

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
        run.log("registered", step=await model.get_step(), attempt=state.attempt,
                gpu_s_prev_attempts=round(state.gpu_s_prev_attempts))
        persist()
        if state.base_val is None:
            state.base_val = await val_eval(run, model, "base")
            persist()
        base = state.base_val
        if cfg.skip_sft:
            sft_stats, sft_step, sft = {"skipped": True}, state.grpo_start_step or 0, base
        else:
            sft_stats = await sft_phase(run, model)
            sft_step = await model.get_step()
            sft = await val_eval(run, model, "sft")
        pool = None
        if cfg.pool_scan:
            if state.pool_ready and (run.out / "pool.json").exists():
                pool = load_pool(run)
                run.log("pool_reused", size=len(pool))
            else:
                pool = await scan_phase(run, model)
                state.pool_ready = True
                persist()
        grpo_info = await grpo_phase(run, model, backend, sft.get("success", 0.0), pool, state, persist)
        grpo = await val_eval(run, model, "grpo")
        persist()
        summary = {"config": asdict(cfg), "val": {"base": base, "sft": sft, "grpo": grpo}, "sft_data": sft_stats,
                   "sft_step": sft_step, "final_step": await model.get_step(), "grpo_val": grpo_info,
                   "best_step": (grpo_info["best"] or {}).get("step"),
                   "model_dir": str(Path(cfg.art_path) / "apishift" / "models" / cfg.run_name),
                   "checkpoints": checkpoint_dirs(cfg), "attempts": state.attempt,
                   "gpu_s_total": round(state.gpu_seconds_total()),
                   "elapsed_s": round(run.elapsed()), "metrics": run.metrics}
        (run.out / "summary.json").write_text(json.dumps(summary, indent=1))
        commit()
        return summary
    finally:
        await backend.close()
