"""Inspect AI task: one sample per (task, mutation) episode of a split.

The solver runs APIShift's own agent loop (apishift.agent.loop) with the
Inspect model as the policy. The scorer reads the reward from the episode
trace. Inspect provides model providers, concurrency, retries and eval logs.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from apishift.envs.mutations import HELDOUT_TYPES, TRAIN_TYPES
from apishift.episode import arun_episode
from apishift.sampling import episodes, sample_mutation
from apishift.tasks import get_task
from harness.policy import InspectPolicy, to_inspect_messages


def group_of(mtype: str) -> str:
    if mtype == "none":
        return "control"
    if mtype in TRAIN_TYPES:
        return "seen"
    if mtype in HELDOUT_TYPES:
        return "heldout"
    raise ValueError(mtype)


def build_dataset(split: str, seed: int, types: tuple[str, ...] | None = None,
                  limit_per_type: int | None = None) -> MemoryDataset:
    counts: dict[str, int] = {}
    samples = []
    for t, spec in episodes(split, seed):
        if types and spec.type not in types:
            continue
        counts[spec.type] = counts.get(spec.type, 0) + 1
        if limit_per_type is not None and counts[spec.type] > limit_per_type:
            continue
        samples.append(Sample(
            id=f"{t.id}__{spec.type}__s{seed}",
            input=t.instruction,
            metadata={"task_id": t.id, "domain": t.domain, "template": t.template,
                      "mutation_type": spec.type, "group": group_of(spec.type), "seed": seed},
        ))
    return MemoryDataset(samples, name=f"apishift-{split}-s{seed}")


@solver
def apishift_agent(policy_name: str = "model", prompt_variant: str = "default"):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        t = get_task(meta["task_id"])
        spec = sample_mutation(t, meta["mutation_type"], meta["seed"])
        if spec is None:
            raise RuntimeError(f"no applicable mutation for {state.sample_id}")
        trace = await arun_episode(t, spec, InspectPolicy(get_model(), policy_name), meta["seed"],
                                   prompt_variant=prompt_variant)
        state.store.set("trace", trace)
        state.messages = to_inspect_messages(trace["messages"])
        state.completed = True
        return state

    return solve


@scorer(metrics={"reward": [mean(), stderr()], "success": [mean(), stderr()],
                 "turns": [mean()], "invalid_calls": [mean()]})
def apishift_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        trace = state.store.get("trace")
        return Score(
            value={"reward": trace["reward"]["total"], "success": float(trace["success"]),
                   "turns": float(trace["turns"]), "invalid_calls": float(trace["invalid_calls"])},
            explanation=f"success={trace['success']} terminated={trace['terminated']}",
            metadata={"group": state.metadata["group"], "mutation_type": state.metadata["mutation_type"]},
        )

    return score


@task
def apishift_eval(split: str = "test", seed: int = 0, types: str | None = None,
                  limit_per_type: int | None = None, policy_name: str = "model",
                  prompt_variant: str = "default") -> Task:
    type_filter = tuple(types.split(",")) if types else None
    return Task(
        dataset=build_dataset(split, seed, type_filter, limit_per_type),
        solver=apishift_agent(policy_name, prompt_variant),
        scorer=apishift_scorer(),
    )
