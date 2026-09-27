"""Isolation tests for resumable run state (train.state.RunState).

Modal restarts preempted GPU functions on the same input; the Sep 27 re-pilot
restarted twice, redid its scan and "base" val on partly trained weights, and
paid ~$3.40 extra. Failure modes, written before the code:
  F1  a restart repeats finished phases (base val, pool scan)
  F2  a restart re-runs finished GRPO steps, so total steps exceed the config
  F3  GPU time across attempts is not accumulated, so restarts bypass the budget
  F4  state from a different config is silently reused
  F5  a crash mid-write leaves a corrupt state file
  F6  val history is lost across a restart
  F7  a fresh run (no state file) is treated as a resumed one
"""

import json

import pytest

from train.state import RunState, StaleState, config_hash

CFG = {"run_name": "r", "base_model": "Qwen/Qwen3-4B", "grpo_steps": 30, "grpo_lr": 3e-6, "pool_scan": True}


def fresh(tmp_path):
    return RunState.load(tmp_path / "state.json", config_hash(CFG))


def test_f7_fresh_run_has_nothing_done(tmp_path):
    s = fresh(tmp_path)
    assert s.base_val is None and not s.pool_ready and s.grpo_start_step is None
    assert s.attempt == 1 and s.remaining_steps(total=30, current_step=0) == 30


def test_f1_finished_phases_survive_restart(tmp_path):
    s = fresh(tmp_path)
    s.base_val = {"success": 0.84}
    s.pool_ready = True
    s.save()
    s2 = fresh(tmp_path)
    assert s2.base_val == {"success": 0.84} and s2.pool_ready and s2.attempt == 2


def test_f2_steps_counted_from_art_step_not_redone(tmp_path):
    s = fresh(tmp_path)
    s.grpo_start_step = 0
    s.save()
    s2 = fresh(tmp_path)  # restarted after ART saved checkpoint 7
    assert s2.remaining_steps(total=30, current_step=7) == 23
    assert s2.remaining_steps(total=30, current_step=30) == 0
    assert s2.remaining_steps(total=30, current_step=35) == 0


def test_f2_grpo_start_step_is_set_once(tmp_path):
    s = fresh(tmp_path)
    s.begin_grpo(current_step=0)
    s.save()
    s2 = fresh(tmp_path)
    s2.begin_grpo(current_step=7)  # restart must not move the origin
    assert s2.grpo_start_step == 0


def test_f3_gpu_time_accumulates_across_attempts(tmp_path):
    s = fresh(tmp_path)
    s.record_gpu(1000)
    s.save()
    s2 = fresh(tmp_path)
    s2.record_gpu(500)
    assert s2.gpu_seconds_total() == 1500
    assert s2.budget_exhausted(limit_s=1400)
    assert not s2.budget_exhausted(limit_s=1600)


def test_f3_record_gpu_is_this_attempts_total_not_increment(tmp_path):
    s = fresh(tmp_path)
    s.record_gpu(100)
    s.record_gpu(250)  # elapsed so far in this attempt
    assert s.gpu_seconds_total() == 250


def test_f4_other_config_is_rejected(tmp_path):
    s = fresh(tmp_path)
    s.save()
    with pytest.raises(StaleState):
        RunState.load(tmp_path / "state.json", config_hash({**CFG, "grpo_lr": 1e-5}))


def test_f4_hash_ignores_key_order_only():
    assert config_hash(dict(reversed(list(CFG.items())))) == config_hash(CFG)
    assert config_hash({**CFG, "grpo_steps": 31}) != config_hash(CFG)


def test_f5_save_is_atomic(tmp_path):
    s = fresh(tmp_path)
    s.save()
    assert not list(tmp_path.glob("*.tmp"))
    json.loads((tmp_path / "state.json").read_text())


def test_f6_history_survives_restart(tmp_path):
    s = fresh(tmp_path)
    s.history.append({"step": 10, "success": 0.85, "reward": 0.8})
    s.save()
    assert fresh(tmp_path).history == [{"step": 10, "success": 0.85, "reward": 0.8}]
