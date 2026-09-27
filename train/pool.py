"""Adaptation-specific difficulty filtering for GRPO (failure modes: tests/isolation/test_pool_selection.py).

From a scan of k samples per training episode, keep (task, mutation type)
pairs where the model:
  - faces an API change (never the no-change control),
  - sometimes succeeds and sometimes fails (lo <= rate <= hi), so GRPO groups have variance,
  - usually solves the same task without the change (control rate >= control_min),
    so the difficulty comes from adapting to the change rather than from the task itself.
Held-out mutation types are never selected.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from apishift.envs.mutations import HELDOUT_TYPES


class PoolTooSmall(RuntimeError):
    pass


def select_pool(traces: Iterable[dict[str, Any]], *, lo: float, hi: float, control_min: float,
                min_size: int) -> list[dict[str, Any]]:
    outcomes: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for t in traces:
        outcomes[(t["task"]["id"], t["mutation"]["type"])].append(bool(t["success"]))

    def rate(key: tuple[str, str]) -> float | None:
        v = outcomes.get(key)
        return sum(v) / len(v) if v else None

    pool = []
    for task_id, mtype in sorted(outcomes):
        if mtype == "none" or mtype in HELDOUT_TYPES:
            continue
        r, control = rate((task_id, mtype)), rate((task_id, "none"))
        if control is None or control < control_min or not (lo <= r <= hi):
            continue
        pool.append({"task_id": task_id, "mutation_type": mtype, "success_rate": round(r, 4),
                     "control_rate": round(control, 4), "samples": len(outcomes[(task_id, mtype)])})
    if len(pool) < min_size:
        raise PoolTooSmall(f"pool has {len(pool)} episodes, need {min_size}")
    return pool
