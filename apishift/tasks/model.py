"""Task data model: instruction, initial state, declarative success expectations, oracle plan."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from apishift.envs.schema import State


@dataclass(frozen=True)
class Unordered:
    """Expected list value where order does not matter."""

    values: tuple[Any, ...]

    def matches(self, actual: Any) -> bool:
        return isinstance(actual, list) and sorted(map(repr, actual)) == sorted(map(repr, self.values))


@dataclass(frozen=True)
class Expect:
    """One expected change between the initial and the final state.

    create: exactly one new record in `collection` whose fields match `fields`.
    update: record `record_id` has `fields`, and it changed nowhere outside
            `fields` or `free_fields`.
    delete: record `record_id` is gone.
    Any change not covered by some Expect fails the predicate.
    """

    collection: str
    kind: Literal["create", "update", "delete"]
    fields: Mapping[str, Any] = field(default_factory=dict)
    record_id: str | None = None
    free_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Call:
    """A canonical API call in an oracle plan. `match` picks the lookup target."""

    endpoint: str
    args: Mapping[str, Any]
    match: Callable[[dict], bool] | None = None


# Oracle plan: yields canonical Calls, receives each canonical response body.
Plan = Callable[[], Generator[Call, dict, None]]


class TargetNotFound(Exception):
    pass


def pick(body: dict, match: Callable[[dict], bool]) -> dict:
    """Pick exactly one matching item from a list response."""
    items = body.get("items", body.get("data", []))
    found = [it for it in items if match(it)]
    if len(found) != 1:
        raise TargetNotFound(f"{len(found)} matches")
    return found[0]


@dataclass(frozen=True, eq=False)
class Task:
    id: str
    domain: str
    template: str
    instance: int
    split: str
    instruction: str
    initial_state: State
    expectations: tuple[Expect, ...]
    plan: Plan

    def check(self, initial: State, final: State) -> bool:
        from apishift.tasks.predicates import check_expectations

        return check_expectations(self.expectations, initial, final)
