"""Isolation tests for task success predicates.

Each test guards one failure mode from docs/failure_modes.md (P1-P10) and runs
over every task. Written before apishift/tasks/ existed.
"""

import copy
import inspect

import pytest

from apishift.tasks import all_tasks, generate_tasks, run_canonical

TASKS = all_tasks()
IDS = [t.id for t in TASKS]


def wrong(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 100 + 1
    if isinstance(value, str):
        return value + "_x"
    if isinstance(value, list):
        return [*value, "intruder@example.com"]
    if value is None:
        return "unexpected"
    if isinstance(value, dict):
        return {**value, "_x": 1}
    raise TypeError(type(value))


def touched_ids(task):
    return {(e.collection, e.record_id) for e in task.expectations if e.record_id}


def added_ids(task, final):
    return {
        (coll, rid)
        for coll, records in final.items()
        for rid in records
        if rid not in task.initial_state.get(coll, {})
    }


@pytest.fixture(scope="module")
def finals():
    return {t.id: run_canonical(t) for t in TASKS}


def test_task_count_and_splits():
    assert 140 <= len(TASKS) <= 160
    assert len(set(IDS)) == len(IDS)
    domains = {t.domain for t in TASKS}
    assert domains == {"calendar", "payments", "ecommerce"}
    assert {t.split for t in TASKS} == {"train", "val", "test"}


def test_p7_predicate_signature_is_state_only():
    params = list(inspect.signature(TASKS[0].check).parameters)
    assert params == ["initial", "final"]


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p1_unchanged_state_fails(task):
    assert not task.check(task.initial_state, copy.deepcopy(task.initial_state))


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p6_correct_final_state_passes(task, finals):
    assert task.check(task.initial_state, finals[task.id])


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p3_collateral_modify_fails(task, finals):
    final = finals[task.id]
    touched = touched_ids(task)
    checked = 0
    for coll, records in task.initial_state.items():
        for rid, rec in records.items():
            if (coll, rid) in touched:
                continue
            field = next(k for k in rec if k != "id")
            bad = copy.deepcopy(final)
            bad[coll][rid][field] = wrong(bad[coll][rid][field])
            assert not task.check(task.initial_state, bad), (coll, rid, field)
            checked += 1
            break
    assert checked > 0


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p3_collateral_delete_fails(task, finals):
    final = finals[task.id]
    touched = touched_ids(task)
    for coll, records in task.initial_state.items():
        for rid in records:
            if (coll, rid) in touched:
                continue
            bad = copy.deepcopy(final)
            del bad[coll][rid]
            assert not task.check(task.initial_state, bad), (coll, rid)
            break


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p3_extra_record_fails(task, finals):
    final = finals[task.id]
    for coll, records in final.items():
        if not records:
            continue
        src = next(iter(records.values()))
        bad = copy.deepcopy(final)
        bad[coll]["zz_extra"] = {**copy.deepcopy(src), "id": "zz_extra"}
        assert not task.check(task.initial_state, bad), coll


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p4_duplicate_side_effect_fails(task, finals):
    final = finals[task.id]
    added = added_ids(task, final)
    for coll, rid in added:
        bad = copy.deepcopy(final)
        bad[coll]["zz_dup"] = {**copy.deepcopy(final[coll][rid]), "id": "zz_dup"}
        assert not task.check(task.initial_state, bad), (coll, rid)


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p5_wrong_expected_value_fails(task, finals):
    final = finals[task.id]
    added = added_ids(task, final)
    checked = 0
    for exp in task.expectations:
        if exp.kind == "update":
            targets = [(exp.collection, exp.record_id)]
        elif exp.kind == "create":
            targets = [(c, rid) for c, rid in added if c == exp.collection]
        else:
            continue
        for coll, rid in targets:
            for field in exp.fields:
                bad = copy.deepcopy(final)
                bad[coll][rid][field] = wrong(bad[coll][rid][field])
                assert not task.check(task.initial_state, bad), (coll, rid, field)
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p9_unrequested_change_on_target_fails(task, finals):
    final = finals[task.id]
    for exp in task.expectations:
        if exp.kind != "update":
            continue
        rec = final[exp.collection][exp.record_id]
        allowed = set(exp.fields) | set(exp.free_fields) | {"id"}
        for field in rec:
            if field in allowed:
                continue
            bad = copy.deepcopy(final)
            bad[exp.collection][exp.record_id][field] = wrong(rec[field])
            assert not task.check(task.initial_state, bad), (exp.record_id, field)


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_p10_free_fields_do_not_cause_false_negative(task, finals):
    final = finals[task.id]
    for exp in task.expectations:
        if exp.kind != "update":
            continue
        for field in exp.free_fields:
            ok = copy.deepcopy(final)
            ok[exp.collection][exp.record_id][field] = "agent chose this"
            assert task.check(task.initial_state, ok), (exp.record_id, field)


def test_p8_running_episodes_does_not_alias_initial_state(finals):
    # finals fixture already ran every canonical plan; initial states must be pristine
    fresh = {t.id: t.initial_state for t in generate_tasks()}
    for task in TASKS:
        assert task.initial_state == fresh[task.id], task.id


def test_task_generation_is_deterministic():
    a = [(t.id, t.instruction, t.initial_state) for t in generate_tasks()]
    b = [(t.id, t.instruction, t.initial_state) for t in generate_tasks()]
    assert a == b
