"""Task registry: 3 domains x 5 templates x 10 instances = 150 tasks.

Split by instance: 0-6 train, 7 val, 8-9 test. Held-out mutation types are
only paired with test tasks (see apishift/sampling.py).
"""

from __future__ import annotations

import json
import random
from functools import lru_cache

from apishift.envs import DOMAINS, LiveEnv
from apishift.envs.mutations import NoMutation
from apishift.envs.schema import State
from apishift.tasks import calendar_tasks, ecommerce_tasks, payments_tasks
from apishift.tasks.model import Call, Expect, TargetNotFound, Task, Unordered, pick

INSTANCES_PER_TEMPLATE = 10
TEMPLATES = {
    "calendar": calendar_tasks.TEMPLATES,
    "payments": payments_tasks.TEMPLATES,
    "ecommerce": ecommerce_tasks.TEMPLATES,
}


def split_for(instance: int) -> str:
    if instance <= 6:
        return "train"
    return "val" if instance == 7 else "test"


def generate_tasks() -> tuple[Task, ...]:
    """Build every task from scratch. Deterministic."""
    tasks = []
    for domain, templates in TEMPLATES.items():
        for name, build in templates.items():
            for i in range(INSTANCES_PER_TEMPLATE):
                task_id = f"{domain}.{name}.{i:02d}"
                instruction, state, expects, plan = build(random.Random(task_id))
                tasks.append(Task(task_id, domain, name, i, split_for(i), instruction, state,
                                  tuple(expects), plan))
    return tuple(tasks)


@lru_cache(maxsize=1)
def all_tasks() -> tuple[Task, ...]:
    return generate_tasks()


def get_task(task_id: str) -> Task:
    return next(t for t in all_tasks() if t.id == task_id)


def run_canonical(task: Task, seed: int = 0) -> State:
    """Execute the oracle plan against the unmutated API and return the final state."""
    env = LiveEnv(DOMAINS[task.domain], task.initial_state, NoMutation(), seed)
    gen = task.plan()
    try:
        call = next(gen)
        while True:
            resp = env.execute(call.endpoint, json.dumps(dict(call.args)))
            if resp["status"] >= 400:
                raise RuntimeError(f"{task.id}: canonical plan failed at {call.endpoint}: {resp}")
            call = gen.send(resp["body"])
    except StopIteration:
        return env.state


__all__ = [
    "Call", "Expect", "Task", "TargetNotFound", "Unordered", "all_tasks", "generate_tasks",
    "get_task", "pick", "run_canonical", "split_for",
]
