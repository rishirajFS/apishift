"""Turn an Inspect eval log into results/*.json and a traces JSONL for later analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog

from apishift.envs.mutations import MUTATION_TYPES
from harness.task import group_of

GROUPS = ("control", "seen", "heldout")


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(float(r[key]) for r in rows) / len(rows), 4)


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_calls = sum(len(r["calls"]) for r in rows)
    return {
        "n": len(rows),
        "success_rate": _rate(rows, "success"),
        "mean_reward": _rate(rows, "reward_total"),
        "mean_turns": _rate(rows, "turns"),
        "invalid_call_rate": round(sum(r["invalid_calls"] for r in rows) / total_calls, 4) if total_calls else None,
    }


def summarize(log: EvalLog, model_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces, errors = [], []
    for sample in log.samples or []:
        trace = sample.store.get("trace") if sample.store else None
        if trace is None:
            errors.append({"id": sample.id, "error": str(sample.error.message if sample.error else "no trace")})
            continue
        traces.append(trace)

    rows = [{**t, "reward_total": t["reward"]["total"]} for t in traces]
    by_group: dict[str, list] = defaultdict(list)
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_group[group_of(r["mutation"]["type"])].append(r)
        by_type[r["mutation"]["type"]].append(r)

    summary = {
        "model": model_key,
        "eval_log": Path(log.location).name if log.location else None,
        "status": log.status,
        "n_samples": len(log.samples or []),
        "n_errors": len(errors),
        "errors": errors[:20],
        "overall": _agg(rows),
        "by_group": {g: _agg(by_group[g]) for g in GROUPS},
        "by_mutation_type": {m: _agg(by_type[m]) for m in MUTATION_TYPES if by_type[m]},
        "terminated": {k: sum(1 for r in rows if r["terminated"] == k) for k in ("final_answer", "max_turns")},
    }
    return summary, traces


def write_results(summary: dict[str, Any], traces: list[dict[str, Any]], out_dir: Path, run_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces").mkdir(exist_ok=True)
    path = out_dir / f"{run_name}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (out_dir / "traces" / f"{run_name}.jsonl").open("w") as fh:
        for t in sorted(traces, key=lambda t: t["episode_id"]):
            fh.write(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n")
    return path
