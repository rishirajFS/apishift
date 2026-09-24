"""Domain definition shared by the three mock APIs, plus small state helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from apishift.envs.errors import not_found
from apishift.envs.schema import Endpoint, State


@dataclass(frozen=True)
class Domain:
    name: str
    description: str
    endpoints: tuple[Endpoint, ...]
    # mutation type -> candidate specs; error_schema_change is derived in sampling
    sites: Mapping[str, tuple[Any, ...]]

    def endpoint_map(self) -> dict[str, Endpoint]:
        return {e.name: e for e in self.endpoints}


def put(state: State, collection: str, record: dict[str, Any]) -> State:
    """Return a new state with `record` inserted or replaced. Never mutates."""
    return {**state, collection: {**state[collection], record["id"]: record}}


def fetch(state: State, collection: str, record_id: str, kind: str, param: str) -> dict[str, Any]:
    rec = state[collection].get(record_id)
    if rec is None:
        raise not_found(kind, record_id, param)
    return rec
