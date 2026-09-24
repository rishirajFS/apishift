"""Run one full episode and produce its JSON trace (messages, calls, reward, mutation)."""

from __future__ import annotations

import json
from typing import Any

from apishift.agent.loop import MAX_TURNS, simulate
from apishift.agent.scripted import OracleAgent
from apishift.agent.types import Policy
from apishift.envs.mutations import Spec
from apishift.reward import compute_reward
from apishift.tasks.model import Task

TRACE_SCHEMA_VERSION = 1


class UnsolvableEpisode(RuntimeError):
    pass


def min_turns_for(task: Task, spec: Spec, seed: int) -> int:
    """Turns the docs-first oracle needs. This is the task minimum under this mutation."""
    result = simulate(task, spec, OracleAgent(task, spec, "docs"), seed)
    if not result.success:
        raise UnsolvableEpisode(f"{task.id} under {spec.to_dict()}")
    return result.turns


def run_episode(task: Task, spec: Spec, policy: Policy, seed: int, max_turns: int = MAX_TURNS) -> dict[str, Any]:
    result = simulate(task, spec, policy, seed, max_turns)
    min_turns = min_turns_for(task, spec, seed)
    reward = compute_reward(success=result.success, invalid_calls=result.invalid_calls,
                            turns=result.turns, min_turns=min_turns)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "episode_id": episode_id(task, spec, seed, policy.name),
        "task": {"id": task.id, "domain": task.domain, "template": task.template,
                 "split": task.split, "instruction": task.instruction},
        "mutation": spec.to_dict(),
        "seed": seed,
        "policy": policy.name,
        "max_turns": max_turns,
        "tools": result.tools,
        "messages": result.messages,
        "calls": result.calls,
        "final_answer": result.final_answer,
        "terminated": result.terminated,
        "turns": result.turns,
        "min_turns": min_turns,
        "invalid_calls": result.invalid_calls,
        "success": result.success,
        "state_changed": result.final_state != task.initial_state,
        "reward": reward.to_dict(),
    }


def episode_id(task: Task, spec: Spec, seed: int, policy_name: str) -> str:
    return f"{task.id}__{spec.type}__s{seed}__{policy_name}"


def dumps_trace(trace: dict[str, Any]) -> str:
    return json.dumps(trace, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
