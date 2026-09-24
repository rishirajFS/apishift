"""Success predicate: evaluate declarative expectations against final state.

It depends only on (initial, final) state and never on model text or the
trace. Failure modes P1-P10 are listed in docs/failure_modes.md.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import permutations
from typing import Any

from apishift.envs.schema import State
from apishift.tasks.model import Expect, Unordered

_MISSING = object()


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Unordered):
        return expected.matches(actual)
    return type(expected) is type(actual) and expected == actual


def _fields_match(fields: dict, record: dict) -> bool:
    return all(k in record and _value_matches(v, record[k]) for k, v in fields.items())


def _creates_match(creates: list[Expect], added: list[dict]) -> bool:
    if len(creates) != len(added):
        return False
    return any(
        all(_fields_match(dict(e.fields), rec) for e, rec in zip(creates, perm, strict=True))
        for perm in permutations(added)
    )


def _updates_match(updates: dict[str, Expect], before: dict, after: dict) -> bool:
    for rid, exp in updates.items():
        old, new = before[rid], after[rid]
        if not _fields_match(dict(exp.fields), new):
            return False
        changed = {k for k in set(old) | set(new) if old.get(k, _MISSING) != new.get(k, _MISSING)}
        if not changed <= set(exp.fields) | set(exp.free_fields):
            return False
    return True


def _check_collection(expects: list[Expect], before: dict, after: dict) -> bool:
    added = [rid for rid in after if rid not in before]
    removed = {rid for rid in before if rid not in after}
    modified = {rid for rid in before if rid in after and before[rid] != after[rid]}

    deletes = {e.record_id for e in expects if e.kind == "delete"}
    updates = {e.record_id: e for e in expects if e.kind == "update"}
    creates = [e for e in expects if e.kind == "create"]

    if removed != deletes or modified != set(updates):
        return False
    if not _updates_match(updates, before, after):
        return False
    return _creates_match(creates, [after[rid] for rid in added])


def check_expectations(expectations: Iterable[Expect], initial: State, final: State) -> bool:
    expects = list(expectations)
    if not expects:
        return False
    for coll in sorted(set(initial) | set(final)):
        before = initial.get(coll, {})
        after = final.get(coll, {})
        if not _check_collection([e for e in expects if e.collection == coll], before, after):
            return False
    return True
