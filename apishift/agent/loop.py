"""Multi-turn agent loop: at most 8 assistant turns; each turn is tool calls or a final answer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from apishift.agent.prompts import system_prompt
from apishift.agent.types import Message, Policy
from apishift.envs import DOMAINS, LiveEnv
from apishift.envs.mutations import Spec
from apishift.envs.schema import State
from apishift.reward import is_invalid_response
from apishift.tasks.model import Task

MAX_TURNS = 8


@dataclass(frozen=True)
class EpisodeResult:
    messages: list[Message]
    tools: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    final_answer: str | None
    turns: int
    invalid_calls: int
    success: bool
    terminated: str
    final_state: State


def tool_content(envelope: dict[str, Any]) -> str:
    return json.dumps(envelope, sort_keys=True, ensure_ascii=False)


def simulate(task: Task, spec: Spec, policy: Policy, seed: int, max_turns: int = MAX_TURNS) -> EpisodeResult:
    env = LiveEnv(DOMAINS[task.domain], task.initial_state, spec, seed)
    tools = env.stale_tools()
    messages: list[Message] = [
        {"role": "system", "content": system_prompt(task.domain)},
        {"role": "user", "content": task.instruction},
    ]
    calls: list[dict[str, Any]] = []
    final_answer: str | None = None
    terminated = "max_turns"
    turns = 0

    for turn in range(1, max_turns + 1):
        action = policy.act(list(messages), tools)
        turns = turn
        messages = [*messages, action.to_message()]
        if not action.tool_calls:
            final_answer = action.content
            terminated = "final_answer"
            break
        for tc in action.tool_calls:
            envelope = env.execute(tc.name, tc.arguments)
            calls = [*calls, {"turn": turn, "id": tc.id, "name": tc.name, "arguments": tc.arguments,
                              "response": envelope, "invalid": is_invalid_response(envelope)}]
            messages = [*messages, {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                    "content": tool_content(envelope)}]

    return EpisodeResult(
        messages=messages,
        tools=tools,
        calls=calls,
        final_answer=final_answer,
        turns=turns,
        invalid_calls=sum(1 for c in calls if c["invalid"]),
        success=task.check(task.initial_state, env.state),
        terminated=terminated,
        final_state=env.state,
    )
