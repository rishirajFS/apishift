"""Multi-turn agent loop: at most 8 assistant turns; each turn is tool calls or a final answer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from apishift.agent.prompts import system_prompt
from apishift.agent.types import AssistantTurn, AsyncPolicy, Message, Policy
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


class Episode:
    """One episode's mutable session: environment, transcript and call log.

    Drive it with step() until done; simulate() and asimulate() are thin loops
    over this for sync (scripted) and async (model-backed) policies.
    """

    def __init__(self, task: Task, spec: Spec, seed: int, max_turns: int = MAX_TURNS,
                 prompt_variant: str = "default") -> None:
        self.task = task
        self.max_turns = max_turns
        self.env = LiveEnv(DOMAINS[task.domain], task.initial_state, spec, seed)
        self.tools = self.env.stale_tools()
        self.messages: list[Message] = [
            {"role": "system", "content": system_prompt(task.domain, prompt_variant)},
            {"role": "user", "content": task.instruction},
        ]
        self.calls: list[dict[str, Any]] = []
        self.final_answer: str | None = None
        self.terminated = "max_turns"
        self.turns = 0

    @property
    def done(self) -> bool:
        return self.terminated == "final_answer" or self.turns >= self.max_turns

    def step(self, action: AssistantTurn) -> None:
        if self.done:
            raise RuntimeError("episode is over")
        self.turns += 1
        self.messages = [*self.messages, action.to_message()]
        if not action.tool_calls:
            self.final_answer = action.content
            self.terminated = "final_answer"
            return
        for tc in action.tool_calls:
            envelope = self.env.execute(tc.name, tc.arguments)
            self.calls = [*self.calls, {"turn": self.turns, "id": tc.id, "name": tc.name,
                                        "arguments": tc.arguments, "response": envelope,
                                        "invalid": is_invalid_response(envelope)}]
            self.messages = [*self.messages, {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                              "content": tool_content(envelope)}]

    def result(self) -> EpisodeResult:
        return EpisodeResult(
            messages=self.messages,
            tools=self.tools,
            calls=self.calls,
            final_answer=self.final_answer,
            turns=self.turns,
            invalid_calls=sum(1 for c in self.calls if c["invalid"]),
            success=self.task.check(self.task.initial_state, self.env.state),
            terminated=self.terminated,
            final_state=self.env.state,
        )


def simulate(task: Task, spec: Spec, policy: Policy, seed: int, max_turns: int = MAX_TURNS,
             prompt_variant: str = "default") -> EpisodeResult:
    ep = Episode(task, spec, seed, max_turns, prompt_variant)
    while not ep.done:
        ep.step(policy.act(list(ep.messages), ep.tools))
    return ep.result()


async def asimulate(
    task: Task, spec: Spec, policy: AsyncPolicy, seed: int, max_turns: int = MAX_TURNS,
    prompt_variant: str = "default",
) -> EpisodeResult:
    ep = Episode(task, spec, seed, max_turns, prompt_variant)
    while not ep.done:
        ep.step(await policy.aact(list(ep.messages), ep.tools))
    return ep.result()
