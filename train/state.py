"""Resumable run state for GPU training that may be preempted and restarted.

Failure modes: tests/isolation/test_run_state.py. The state file lives next to
the run's outputs on the checkpoints Volume. ART's checkpoint step is
authoritative for how many GRPO steps are done; this file records what ART does
not know: finished phases, where GRPO started, val history, and GPU time spent
across all attempts (so restarts cannot bypass the budget).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class StaleState(RuntimeError):
    pass


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class RunState:
    path: Path
    config_hash: str
    attempt: int = 1
    base_val: dict[str, Any] | None = None
    pool_ready: bool = False
    grpo_start_step: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    gpu_s_prev_attempts: float = 0.0
    gpu_s_this_attempt: float = 0.0

    @classmethod
    def load(cls, path: Path, cfg_hash: str) -> RunState:
        path = Path(path)
        if not path.exists():
            return cls(path=path, config_hash=cfg_hash)
        data = json.loads(path.read_text())
        if data["config_hash"] != cfg_hash:
            raise StaleState(f"{path} belongs to a different config ({data['config_hash']} != {cfg_hash})")
        data.pop("path", None)
        state = cls(path=path, **data)
        # a new attempt: fold the previous attempt's GPU time into the running total
        state.attempt += 1
        state.gpu_s_prev_attempts += state.gpu_s_this_attempt
        state.gpu_s_this_attempt = 0.0
        return state

    def save(self) -> None:
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self.path)

    def begin_grpo(self, current_step: int) -> None:
        if self.grpo_start_step is None:
            self.grpo_start_step = current_step

    def remaining_steps(self, total: int, current_step: int) -> int:
        start = self.grpo_start_step if self.grpo_start_step is not None else current_step
        return max(0, total - (current_step - start))

    def record_gpu(self, elapsed_this_attempt_s: float) -> None:
        self.gpu_s_this_attempt = float(elapsed_this_attempt_s)

    def gpu_seconds_total(self) -> float:
        return self.gpu_s_prev_attempts + self.gpu_s_this_attempt

    def budget_exhausted(self, limit_s: float) -> bool:
        return self.gpu_seconds_total() > limit_s
