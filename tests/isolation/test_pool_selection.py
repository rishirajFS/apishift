"""Isolation tests for adaptation-specific difficulty filtering (train.pool.select_pool).

Failure modes, written before the code:
  F1  no-change control episodes enter the pool (they carry no adaptation signal)
  F2  episodes the model always solves or always fails enter the pool (no GRPO variance)
  F3  episodes from tasks the model cannot do even without a change enter the pool
      (difficulty unrelated to adaptation, e.g. misreading an optional filter)
  F4  held-out mutation types enter the pool (would leak the test)
  F5  bounds are exclusive where they should be inclusive (10% / 90% edges dropped)
  F6  an empty or too-small pool is returned silently instead of raising
  F7  the pool order depends on dict iteration or input order (non-reproducible runs)
"""

import pytest

from train.pool import PoolTooSmall, select_pool


def tr(task, mtype, success):
    return {"task": {"id": task}, "mutation": {"type": mtype}, "success": success}


def scan(task, mtype, successes):
    return [tr(task, mtype, s) for s in successes]


BASE = (
    scan("t1", "none", [1, 1, 1, 1])
    + scan("t1", "rename_param", [1, 0, 1, 0])        # mixed, control ok -> in
    + scan("t1", "format_change", [1, 1, 1, 1])       # always solved -> out (F2)
    + scan("t1", "deprecation_with_migration", [0, 0, 0, 0])  # never solved -> out (F2)
    + scan("t2", "none", [0, 0, 0, 1])                # control 25% -> task excluded (F3)
    + scan("t2", "rename_param", [1, 0, 0, 1])
    + scan("t3", "none", [1, 1, 0, 1])
    + scan("t3", "new_required_field", [0, 0, 0, 1])  # 25%, control 75% -> in
    + scan("t3", "pagination_change", [1, 0, 1, 0])   # held-out -> out (F4)
)


def keys(pool):
    return [(p["task_id"], p["mutation_type"]) for p in pool]


def test_selects_only_mixed_adaptation_episodes():
    pool = select_pool(BASE, lo=0.1, hi=0.9, control_min=0.5, min_size=1)
    assert keys(pool) == [("t1", "rename_param"), ("t3", "new_required_field")]


def test_f1_control_never_in_pool():
    pool = select_pool(BASE + scan("t4", "none", [1, 0, 1, 0]), lo=0.1, hi=0.9, control_min=0.0, min_size=1)
    assert all(p["mutation_type"] != "none" for p in pool)


def test_f3_control_threshold_applies_per_task():
    pool = select_pool(BASE, lo=0.1, hi=0.9, control_min=0.0, min_size=1)
    assert ("t2", "rename_param") in keys(pool)
    pool = select_pool(BASE, lo=0.1, hi=0.9, control_min=0.5, min_size=1)
    assert ("t2", "rename_param") not in keys(pool)


def test_f3_missing_control_scan_excludes_task():
    pool = select_pool(scan("t9", "rename_param", [1, 0]), lo=0.1, hi=0.9, control_min=0.5, min_size=0)
    assert pool == []


@pytest.mark.parametrize("held_out", ["pagination_change", "error_schema_change"])
def test_f4_heldout_never_in_pool(held_out):
    data = scan("t5", "none", [1, 1]) + scan("t5", held_out, [1, 0])
    assert select_pool(data, lo=0.0, hi=1.0, control_min=0.0, min_size=0) == []


def test_f5_bounds_inclusive():
    data = scan("t6", "none", [1] * 10) + scan("t6", "rename_param", [1] + [0] * 9) \
        + scan("t7", "none", [1] * 10) + scan("t7", "format_change", [1] * 9 + [0])
    assert len(select_pool(data, lo=0.1, hi=0.9, control_min=0.5, min_size=1)) == 2


def test_f6_too_small_raises():
    with pytest.raises(PoolTooSmall):
        select_pool(BASE, lo=0.1, hi=0.9, control_min=0.5, min_size=3)


def test_f7_order_is_input_independent():
    a = select_pool(BASE, lo=0.1, hi=0.9, control_min=0.5, min_size=1)
    b = select_pool(list(reversed(BASE)), lo=0.1, hi=0.9, control_min=0.5, min_size=1)
    assert a == b


def test_pool_entries_carry_rates():
    [first, _] = select_pool(BASE, lo=0.1, hi=0.9, control_min=0.5, min_size=1)
    assert first == {"task_id": "t1", "mutation_type": "rename_param", "success_rate": 0.5,
                     "control_rate": 1.0, "samples": 4}
