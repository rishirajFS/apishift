"""Seeded mutation sampling and episode enumeration.

sample_mutation(task, type, seed) shuffles the domain's candidate sites with
an RNG derived from (task, type, seed). It returns the first candidate that is
  - non-inert: the stale plan fails under it (E1), and
  - solvable: the docs-first oracle succeeds under it (E2).
If no candidate qualifies, the (task, type) pair is not applicable and it
returns None.
"""

from __future__ import annotations

import hashlib
import random
from functools import cache

from apishift.agent.loop import simulate
from apishift.agent.scripted import OracleAgent, StaleAgent
from apishift.envs import DOMAINS
from apishift.envs.mutations import (
    DIVERSITY_TYPES,
    HELDOUT_TYPES,
    MUTATION_TYPES,
    SEEN_EVAL_TYPES,
    TEST_TYPES,
    TRAIN_TYPES,
    Compound,
    ErrorSchemaChange,
    NewRequiredField,
    NoMutation,
    RenameParam,
    Spec,
)
from apishift.tasks import all_tasks, get_task
from apishift.tasks.model import Task

SPLIT_TYPES = {
    "train": ("none", *TRAIN_TYPES),
    "val": ("none", *TRAIN_TYPES),
    "test": TEST_TYPES,  # frozen: diversity types are training-only
}


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def candidate_specs(domain: str, mtype: str) -> tuple[Spec, ...]:
    sites = DOMAINS[domain].sites
    if mtype == "none":
        return (NoMutation(),)
    if mtype == "error_schema_change":
        inner = [s for s in (*sites["rename_param"], *sites["new_required_field"])
                 if isinstance(s, (RenameParam, NewRequiredField))]
        return tuple(ErrorSchemaChange(s) for s in inner)
    if mtype == "compound_change":
        return compound_candidates(domain)
    return tuple(sites.get(mtype, ()))


MAX_COMPOUND_CANDIDATES = 40


def compound_candidates(domain: str) -> tuple[Spec, ...]:
    """Pairs of seen-type changes of different types, a stable sample per domain."""
    sites = DOMAINS[domain].sites
    singles = [(t, s) for t in SEEN_EVAL_TYPES for s in sites.get(t, ())]
    pairs = [Compound(a, b) for i, (ta, a) in enumerate(singles) for tb, b in singles[i + 1:] if ta != tb]
    _rng("compound", domain).shuffle(pairs)
    return tuple(pairs[:MAX_COMPOUND_CANDIDATES])


def is_valid_for(task: Task, spec: Spec, seed: int) -> bool:
    if isinstance(spec, NoMutation):
        return True
    stale = simulate(task, spec, StaleAgent(task), seed)
    if stale.success:
        return False
    return simulate(task, spec, OracleAgent(task, spec, "docs"), seed).success


@cache
def _sample(task_id: str, mtype: str, seed: int) -> Spec | None:
    task = get_task(task_id)
    candidates = list(candidate_specs(task.domain, mtype))
    _rng(task_id, mtype, seed).shuffle(candidates)
    return next((s for s in candidates if is_valid_for(task, s, seed)), None)


def sample_mutation(task: Task, mtype: str, seed: int) -> Spec | None:
    if mtype not in MUTATION_TYPES:
        raise ValueError(f"unknown mutation type: {mtype}")
    if mtype in HELDOUT_TYPES and task.split != "test":
        raise ValueError(f"held-out type {mtype} requested for {task.split} task {task.id}")
    if mtype in DIVERSITY_TYPES and task.split == "test":
        raise ValueError(f"training-only type {mtype} requested for test task {task.id}")
    return _sample(task.id, mtype, seed)


def episodes(split: str, seed: int) -> list[tuple[Task, Spec]]:
    """Every applicable (task, spec) pair for a split, in a stable order."""
    out = []
    for task in all_tasks():
        if task.split != split:
            continue
        for mtype in SPLIT_TYPES[split]:
            spec = sample_mutation(task, mtype, seed)
            if spec is not None:
                out.append((task, spec))
    return out
