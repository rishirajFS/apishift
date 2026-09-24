"""Episode reward. Pure function of outcome counts; never sees model text.

reward = 1.0 if success, minus 0.05 per invalid call, minus 0.02 per turn
beyond the task's minimum, floored at 0. Failure modes R1-R12 are listed in
docs/failure_modes.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SUCCESS_REWARD = 1.0
INVALID_CALL_PENALTY = 0.05
TURN_PENALTY = 0.02
_PRECISION = 4


@dataclass(frozen=True)
class Reward:
    total: float
    success: float
    invalid_penalty: float
    turn_penalty: float

    def to_dict(self) -> dict[str, float]:
        return {"total": self.total, "success": self.success,
                "invalid_penalty": self.invalid_penalty, "turn_penalty": self.turn_penalty}


def compute_reward(*, success: bool, invalid_calls: int, turns: int, min_turns: int) -> Reward:
    if invalid_calls < 0 or turns < 0 or min_turns < 0:
        raise ValueError("counts must be non-negative")
    base = SUCCESS_REWARD if success else 0.0
    invalid_pen = round(INVALID_CALL_PENALTY * invalid_calls, _PRECISION)
    turn_pen = round(TURN_PENALTY * max(0, turns - min_turns), _PRECISION)
    total = round(max(0.0, base - invalid_pen - turn_pen), _PRECISION) if success else 0.0
    return Reward(total=total, success=base, invalid_penalty=invalid_pen, turn_penalty=turn_pen)


def is_invalid_response(envelope: Mapping[str, Any]) -> bool:
    """A tool call is invalid if the API answered with an error (any envelope shape)."""
    status = envelope.get("status")
    return not isinstance(status, int) or status >= 400
