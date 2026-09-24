"""Isolation tests for the reward function.

Each test guards one failure mode from docs/failure_modes.md (R1-R12).
Written before apishift/reward.py existed.
"""

import inspect
import itertools
import json

import pytest

from apishift.reward import (
    INVALID_CALL_PENALTY,
    SUCCESS_REWARD,
    TURN_PENALTY,
    compute_reward,
    is_invalid_response,
)

GRID = list(
    itertools.product(
        (True, False),  # success
        range(0, 12),  # invalid_calls
        range(1, 12),  # turns
        range(1, 8),  # min_turns
    )
)


def r(success, invalid_calls=0, turns=3, min_turns=3):
    return compute_reward(
        success=success, invalid_calls=invalid_calls, turns=turns, min_turns=min_turns
    ).total


def test_constants_match_spec():
    assert SUCCESS_REWARD == 1.0
    assert INVALID_CALL_PENALTY == 0.05
    assert TURN_PENALTY == 0.02


@pytest.mark.parametrize("success,invalid,turns,min_turns", GRID)
def test_r1_reward_bounded(success, invalid, turns, min_turns):
    total = r(success, invalid, turns, min_turns)
    assert 0.0 <= total <= 1.0


@pytest.mark.parametrize("invalid,turns", list(itertools.product(range(0, 5), range(1, 9))))
def test_r2_failure_never_positive(invalid, turns):
    assert r(False, invalid, turns, min_turns=3) == 0.0


def test_r3_invalid_calls_penalized():
    assert r(True, invalid_calls=0) == 1.0
    assert r(True, invalid_calls=1) == pytest.approx(0.95)
    assert r(True, invalid_calls=3) == pytest.approx(0.85)


def test_r4_valid_responses_not_invalid():
    assert not is_invalid_response({"status": 200, "body": {"items": []}})
    assert not is_invalid_response({"status": 201, "body": {"id": "evt_1"}})
    docs = {"status": 200, "body": {"endpoint": "create_event", "parameters": {}}}
    assert not is_invalid_response(docs)


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_r5_no_turn_penalty_within_minimum(turns):
    assert r(True, turns=turns, min_turns=3) == 1.0


def test_r6_turn_penalty_beyond_minimum():
    assert r(True, turns=4, min_turns=3) == pytest.approx(0.98)
    assert r(True, turns=8, min_turns=3) == pytest.approx(0.90)


def test_r6_penalties_combine():
    assert r(True, invalid_calls=2, turns=5, min_turns=3) == pytest.approx(0.86)


def test_r7_floor_at_zero():
    assert r(True, invalid_calls=40, turns=8, min_turns=3) == 0.0


@pytest.mark.parametrize("success,invalid,turns,min_turns", GRID)
def test_r8_monotone(success, invalid, turns, min_turns):
    base = r(success, invalid, turns, min_turns)
    assert r(success, invalid + 1, turns, min_turns) <= base
    assert r(success, invalid, turns + 1, min_turns) <= base
    if not success:
        assert r(True, invalid, turns, min_turns) >= base


def test_r10_reward_signature_has_no_text_input():
    params = set(inspect.signature(compute_reward).parameters)
    assert params == {"success", "invalid_calls", "turns", "min_turns"}


def test_r9_breakdown_components_consistent():
    br = compute_reward(success=True, invalid_calls=1, turns=5, min_turns=3)
    assert br.success == 1.0
    assert br.invalid_penalty == pytest.approx(0.05)
    assert br.turn_penalty == pytest.approx(0.04)
    assert br.total == pytest.approx(0.91)


@pytest.mark.parametrize(
    "envelope",
    [
        {"status": 400, "error": {"code": "parameter_unknown", "message": "x"}},
        {"status": 400, "error": {"code": "malformed_arguments", "message": "x"}},
        {"status": 404, "error": {"code": "unknown_endpoint", "message": "x"}},
        {"status": 409, "error": {"code": "conflict", "message": "x"}},
        {"status": 410, "error": {"code": "endpoint_removed", "message": "x"}},
        # held-out error envelope (RFC 7807 style) must still count as invalid
        {"status": 422, "type": "about:blank", "title": "Unprocessable", "detail": "x"},
    ],
)
def test_r11_error_responses_are_invalid(envelope):
    assert is_invalid_response(envelope)


def test_r11_envelope_without_status_is_invalid():
    assert is_invalid_response({"body": {}})


@pytest.mark.parametrize("success,invalid,turns,min_turns", GRID[:200])
def test_r12_reward_serializes_stably(success, invalid, turns, min_turns):
    total = r(success, invalid, turns, min_turns)
    assert json.dumps(total) == json.dumps(round(total, 4))
