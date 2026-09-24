"""Reproducible E2E run: every task x every applicable mutation x scripted agents.

    uv run python -m apishift.e2e --seed 0 --out artifacts/e2e

Writes:
    sweep.json          per-episode outcomes for every agent, coverage, summary
    traces/*.json       full traces for one showcase task per domain x every type
    MANIFEST.sha256     sha256 of every file above; byte-identical for a given seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from apishift.agent.scripted import (
    ClaimSuccessAgent,
    DocsOnlyAgent,
    MalformedArgsAgent,
    OracleAgent,
    StaleAgent,
    UnknownToolAgent,
)
from apishift.envs import DOMAINS, LiveEnv
from apishift.envs.mutations import MUTATION_TYPES, NoMutation, Spec
from apishift.episode import dumps_trace, run_episode
from apishift.sampling import episodes
from apishift.tasks.model import Task

AGENTS: dict[str, Callable[[Task, Spec], Any]] = {
    "stale": lambda t, s: StaleAgent(t),
    "oracle_docs": lambda t, s: OracleAgent(t, s, "docs"),
    "oracle_error": lambda t, s: OracleAgent(t, s, "error"),
    "claim_success": lambda t, s: ClaimSuccessAgent(t),
    "docs_only": lambda t, s: DocsOnlyAgent(t),
    "malformed_args": lambda t, s: MalformedArgsAgent(t),
    "unknown_tool": lambda t, s: UnknownToolAgent(t),
}
TRACED_AGENTS = ("stale", "oracle_docs", "oracle_error")
SPLITS = ("train", "val", "test")


def docs_reflect_live(task: Task, spec: Spec, seed: int) -> bool:
    if isinstance(spec, NoMutation):
        return True
    env = LiveEnv(DOMAINS[task.domain], task.initial_state, spec, seed)
    affected = spec.affected_endpoints(DOMAINS[task.domain].endpoint_map())
    if not affected:
        return False
    for name in affected:
        live = {k: v for k, v in env.docs_for(name).items() if k != "error_format"}
        if live == env.stale_docs_for(name):
            return False
    return True


def _outcome(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": trace["success"],
        "reward": trace["reward"]["total"],
        "turns": trace["turns"],
        "invalid_calls": trace["invalid_calls"],
        "state_unchanged": not trace["state_changed"],
    }


def _showcase(pairs: list[tuple[Task, Spec]]) -> set[str]:
    """First test task per domain that has every mutation type applicable."""
    types_by_task: dict[str, set[str]] = defaultdict(set)
    for task, spec in pairs:
        types_by_task[task.id].add(spec.type)
    chosen: dict[str, str] = {}
    for task, _ in pairs:
        if task.split == "test" and task.domain not in chosen and types_by_task[task.id] == set(MUTATION_TYPES):
            chosen[task.domain] = task.id
    return set(chosen.values())


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["mutation"]["type"]].append(row)
    out = {}
    for mtype in MUTATION_TYPES:
        group = by_type.get(mtype, [])
        if not group:
            continue
        out[mtype] = {"episodes": len(group)} | {
            agent: {
                "success_rate": round(sum(r["agents"][agent]["success"] for r in group) / len(group), 4),
                "mean_reward": round(sum(r["agents"][agent]["reward"] for r in group) / len(group), 4),
            }
            for agent in ("stale", "oracle_docs", "oracle_error")
        }
    return out


def run(seed: int, out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for old in traces_dir.glob("*.json"):
        old.unlink()

    pairs = [pair for split in SPLITS for pair in episodes(split, seed)]
    showcase = _showcase(pairs)
    rows = []
    coverage: dict[str, dict[str, dict[str, int]]] = {
        s: {d: {m: 0 for m in MUTATION_TYPES} for d in DOMAINS} for s in SPLITS
    }
    for task, spec in pairs:
        coverage[task.split][task.domain][spec.type] += 1
        agents = {}
        min_turns = None
        for name, make in AGENTS.items():
            trace = run_episode(task, spec, make(task, spec), seed)
            min_turns = trace["min_turns"]
            agents[name] = _outcome(trace)
            if task.id in showcase and name in TRACED_AGENTS:
                (traces_dir / f"{trace['episode_id']}.json").write_text(dumps_trace(trace))
        rows.append({
            "task_id": task.id, "domain": task.domain, "template": task.template, "split": task.split,
            "mutation": spec.to_dict(), "min_turns": min_turns,
            "docs_reflect_live": docs_reflect_live(task, spec, seed), "agents": agents,
        })

    sweep = {"seed": seed, "n_episodes": len(rows), "summary": _summary(rows),
             "coverage": coverage, "showcase_tasks": sorted(showcase), "episodes": rows}
    (out_dir / "sweep.json").write_text(json.dumps(sweep, sort_keys=True, indent=1) + "\n")
    _write_manifest(out_dir)
    return sweep


def _write_manifest(out_dir: Path) -> None:
    lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/e2e"))
    args = parser.parse_args()
    sweep = run(args.seed, args.out)
    print(f"{sweep['n_episodes']} episodes -> {args.out}")
    for mtype, row in sweep["summary"].items():
        print(f"  {mtype:28s} n={row['episodes']:4d}  stale={row['stale']['success_rate']:.2f}  "
              f"oracle_docs={row['oracle_docs']['success_rate']:.2f}  "
              f"oracle_error={row['oracle_error']['success_rate']:.2f}")


if __name__ == "__main__":
    main()
