"""Scripted policies used to validate the environment end to end.

StaleAgent       runs the task's canonical plan unchanged (what an agent that
                 trusts its stale schemas does). It must succeed with no
                 mutation and fail under every mutation (non-inertness).
OracleAgent      runs the plan through the mutation's inverse mapping (it reads
                 the spec, so it is an oracle, not a policy). discover="docs"
                 reads docs first; this defines min_turns. discover="error"
                 probes with the stale call first.
ClaimSuccess / MalformedArgs / UnknownTool / DocsOnly: adversarial agents
                 that must earn 0 reward and leave state unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from typing import Any

from apishift.agent.types import AssistantTurn, Message, ToolCall
from apishift.envs.live import DOCS_TOOL_NAME
from apishift.envs.mutations import PaginationChange, Spec
from apishift.tasks.model import Call, TargetNotFound, Task

# A live plan yields (tool name, args) and receives the response envelope.
LivePlan = Generator[tuple[str, dict[str, Any]], dict[str, Any], str]


class _PlanAgent:
    """Adapts a generator plan to the Policy interface, one tool call per turn."""

    name = "plan"

    def __init__(self, make_plan: Callable[[], LivePlan]) -> None:
        self._make_plan = make_plan
        self._plan: LivePlan | None = None
        self._n = 0

    def act(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        try:
            if self._plan is None:
                self._plan = self._make_plan()
                name, args = next(self._plan)
            else:
                name, args = self._plan.send(json.loads(messages[-1]["content"]))
        except StopIteration as stop:
            return AssistantTurn(content=stop.value or "Done.")
        self._n += 1
        call = ToolCall(id=f"call_{self._n}", name=name, arguments=json.dumps(args, sort_keys=True))
        return AssistantTurn(content="", tool_calls=(call,))


def _run_canonical_steps(task: Task, step: Callable[[Call], Generator]) -> LivePlan:
    canon = task.plan()
    try:
        call = next(canon)
        while True:
            body = yield from step(call)
            call = canon.send(body)
    except StopIteration:
        return "Done."
    except (TargetNotFound, _GiveUp, KeyError, TypeError) as exc:  # KeyError: a response field was renamed
        return f"I could not complete the task: {exc}"


class _GiveUp(Exception):
    pass


def stale_plan(task: Task) -> LivePlan:
    def step(call: Call):
        env = yield (call.endpoint, dict(call.args))
        if env.get("status", 500) >= 400:
            raise _GiveUp(f"{call.endpoint} returned {env.get('status')}")
        return env["body"]

    return (yield from _run_canonical_steps(task, step))


def oracle_plan(task: Task, spec: Spec, discover: str) -> LivePlan:
    discovered: set[str] = set()

    def step(call: Call):
        # one docs read (or stale probe) per change still unknown at this call; a compound change
        # spread over two endpoints needs two
        pending = spec.discovery_keys(call.endpoint, dict(call.args)) - discovered
        if pending:
            discovered.update(pending)
            if discover == "docs":
                yield (DOCS_TOOL_NAME, {"endpoint": call.endpoint})
            else:
                yield (call.endpoint, dict(call.args))
        if isinstance(spec, PaginationChange) and call.endpoint == spec.endpoint:
            return (yield from _paginate(call))
        name, args = spec.oracle_request(call.endpoint, dict(call.args))
        env = yield (name, args)
        if env.get("status", 500) >= 400:
            raise _GiveUp(f"oracle call {name} failed: {env}")
        return spec.oracle_response(env["body"])

    def _paginate(call: Call):
        items: list[dict] = []
        cursor = None
        while True:
            args = {**call.args, **({"cursor": cursor} if cursor else {})}
            env = yield (call.endpoint, args)
            if env.get("status", 500) >= 400:
                raise _GiveUp(f"pagination failed: {env}")
            page = env["body"]
            items = [*items, *page["data"]]
            if call.match is not None and any(call.match(it) for it in page["data"]):
                break
            if not page["has_more"]:
                break
            cursor = page["next_cursor"]
        return {"items": items}

    return (yield from _run_canonical_steps(task, step))


class StaleAgent(_PlanAgent):
    name = "stale"

    def __init__(self, task: Task) -> None:
        super().__init__(lambda: stale_plan(task))


class OracleAgent(_PlanAgent):
    def __init__(self, task: Task, spec: Spec, discover: str = "docs") -> None:
        if discover not in ("docs", "error"):
            raise ValueError(discover)
        self.name = f"oracle_{discover}"
        super().__init__(lambda: oracle_plan(task, spec, discover))


class _FixedAgent(_PlanAgent):
    def __init__(self, steps: list[tuple[str, dict[str, Any] | str]], final: str) -> None:
        def plan() -> LivePlan:
            for name, args in steps:  # noqa: UP028 - plan must accept .send()
                yield (name, args)  # type: ignore[misc]
            return final

        super().__init__(plan)


class ClaimSuccessAgent(_FixedAgent):
    name = "claim_success"

    def __init__(self, task: Task) -> None:
        super().__init__([], "I have completed the task successfully. Everything is done.")


class DocsOnlyAgent(_FixedAgent):
    name = "docs_only"

    def __init__(self, task: Task) -> None:
        first = task.plan().send(None).endpoint
        super().__init__([(DOCS_TOOL_NAME, {}), (DOCS_TOOL_NAME, {"endpoint": first})],
                         "Task complete.")


class MalformedArgsAgent:
    name = "malformed_args"

    def __init__(self, task: Task) -> None:
        self._endpoint = task.plan().send(None).endpoint
        self._done = False

    def act(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        if self._done:
            return AssistantTurn(content="Task complete.")
        self._done = True
        bad = ToolCall(id="call_1", name=self._endpoint, arguments='{"oops": ')
        return AssistantTurn(content="", tool_calls=(bad,))


class UnknownToolAgent(_FixedAgent):
    name = "unknown_tool"

    def __init__(self, task: Task) -> None:
        super().__init__([("delete_all_records", {"confirm": True})], "Task complete.")
