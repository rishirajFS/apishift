# Failure modes: reward and success predicates

Per AGENTS.md, the reward function and the task success checks are the only
components tested in isolation. This file lists every way they could fail. It
was written before any implementation. Each failure mode has an ID, and the
test that guards it carries the same ID.

Tests: `tests/isolation/test_reward_failure_modes.py` (R*),
`tests/isolation/test_predicate_failure_modes.py` (P*). The episode-level
modes (E*) are checked by the E2E sweep in `tests/e2e/test_episode_e2e.py`.

## Reward (`apishift/reward.py`)

Contract: `compute_reward(success, invalid_calls, turns, min_turns)`.
reward = 1.0 if success, minus 0.05 per invalid call, minus 0.02 per turn
beyond `min_turns`, floored at 0.

| ID  | Failure mode |
|-----|--------------|
| R1  | Reward leaves [0, 1] for some input (negative, or above 1.0). |
| R2  | A failed episode gets positive reward, e.g. penalties are applied to a base of 1.0 regardless of success. |
| R3  | Invalid calls are not penalized, or they increase the reward. |
| R4  | Valid calls, including `get_api_docs`, are counted as invalid. |
| R5  | The turn penalty is applied at or below `min_turns`. |
| R6  | The turn penalty is not applied beyond `min_turns`, or has the wrong per-turn size. |
| R7  | No floor: many invalid calls push the reward below 0. |
| R8  | Non-monotone: adding an invalid call or a turn raises the reward. |
| R9  | The final-answer text affects the reward, e.g. an agent that only says "Task complete" scores. |
| R10 | The reward accepts model text as input at all. This is a structural guard for R9. |
| R11 | Invalid-call classification is wrong: a 4xx/410/malformed-args response is not counted, or a 2xx is counted. This includes the held-out error envelope (`error_schema_change`), which must still count as invalid. |
| R12 | Floating-point drift makes identical episodes serialize to different rewards. |

## Success predicates (`apishift/tasks/predicates.py`)

Contract: `task.check(initial_state, final_state) -> bool`. It depends only on
the two states and never on model text or the trace.

| ID  | Failure mode |
|-----|--------------|
| P1  | The predicate passes on the unchanged initial state. |
| P2  | The predicate passes when the action hit a distractor record instead of the target. Covered by P1 (target unchanged) plus P3 (distractor changed). |
| P3  | Collateral damage passes: an unrelated record was modified or deleted, or an extra record was created. |
| P4  | Duplicate side effects pass, e.g. two charges or two events created where one was asked for. |
| P5  | A wrong value in an expected field passes, e.g. the amount is off by 100x (cents vs dollars), the time is wrong, or the attendee is wrong. |
| P6  | False negative: the correct final state fails. Guarded by the canonical oracle's final state passing for every task, and by the E2E oracle sweep. |
| P7  | The predicate accepts model text or the trace. This is a structural signature check. |
| P8  | Aliasing: running an episode mutates the task's stored initial state, so later episodes start from, or are judged against, a corrupted state. |
| P9  | An unrequested change to an updated record passes, e.g. a reschedule that also renames the event. |
| P10 | Fields the agent may legitimately choose freely, such as an optional cancellation reason, cause a false negative. |

## Episode level (E2E sweep)

| ID  | Failure mode |
|-----|--------------|
| E1  | The mutation is inert: the stale (unadapted) plan still succeeds. |
| E2  | The mutation is unsolvable: an agent that adapts correctly still fails. |
| E3  | `get_api_docs` shows the stale schema instead of the live one. |
| E4  | Non-determinism: the same seed gives different traces or bytes. |
| E5  | `get_api_docs`, unknown tools, or malformed calls change state. |
| E6  | Held-out mutation types leak into the train or val splits. |
| E7  | Malformed tool arguments crash the loop instead of counting as invalid. |
| E8  | An agent that only claims success, with no tool calls, gets reward. |
| E9  | The no-mutation control is broken: the stale plan fails with no mutation. |
