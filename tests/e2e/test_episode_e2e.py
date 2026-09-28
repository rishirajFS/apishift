"""End-to-end episode test.

Runs full episodes (task loaded, mutation applied by seed, agent loop, reward)
with scripted agents over every task and every applicable mutation type. It
writes the artifact to artifacts/e2e/ and checks that artifact byte for byte
against a second run and against the committed copy.

Regenerate the committed artifact with: make e2e
"""

import hashlib
import json
from pathlib import Path

import pytest

from apishift.e2e import run
from apishift.envs.mutations import (
    DIVERSITY_TYPES,
    HELDOUT_TYPES,
    MUTATION_TYPES,
    SEEN_EVAL_TYPES,
    TEST_TYPES,
    TRAIN_TYPES,
)

REPO = Path(__file__).resolve().parents[2]
COMMITTED = REPO / "artifacts" / "e2e"
SEED = 0


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    a = tmp_path_factory.mktemp("run_a")
    b = tmp_path_factory.mktemp("run_b")
    run(seed=SEED, out_dir=a)
    run(seed=SEED, out_dir=b)
    return a, b


@pytest.fixture(scope="module")
def sweep(runs):
    return json.loads((runs[0] / "sweep.json").read_text())


def test_e4_byte_identical_across_runs(runs):
    a, b = runs
    ta, tb = tree_bytes(a), tree_bytes(b)
    assert ta.keys() == tb.keys()
    diff = [k for k in ta if ta[k] != tb[k]]
    assert not diff, f"non-deterministic files: {diff[:5]}"


def test_e4_matches_committed_artifact(runs):
    manifest = COMMITTED / "MANIFEST.sha256"
    assert manifest.exists(), "no committed artifact; run `make e2e`"
    fresh = (runs[0] / "MANIFEST.sha256").read_text()
    assert fresh == manifest.read_text(), "artifact drifted; inspect diff then `make e2e`"


def test_manifest_hashes_are_correct(runs):
    root = runs[0]
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


def test_traces_written_with_reward_and_mutation(runs):
    traces = sorted((runs[0] / "traces").glob("*.json"))
    assert len(traces) >= 3 * len(TEST_TYPES)
    for path in traces:
        tr = json.loads(path.read_text())
        for key in ("messages", "calls", "reward", "mutation", "seed", "task", "min_turns"):
            assert key in tr, (path.name, key)
        assert tr["seed"] == SEED
        assert tr["turns"] <= 8
        for call in tr["calls"]:
            assert "response" in call and "status" in call["response"]


def episodes_of(sweep, mtype):
    return [e for e in sweep["episodes"] if e["mutation"]["type"] == mtype]


def test_e9_no_mutation_control_stale_plan_succeeds(sweep):
    eps = episodes_of(sweep, "none")
    assert len(eps) == len({e["task_id"] for e in sweep["episodes"]})
    for e in eps:
        assert e["agents"]["stale"]["success"], e["task_id"]
        assert e["agents"]["stale"]["reward"] == 1.0, e["task_id"]


@pytest.mark.parametrize("mtype", TRAIN_TYPES + HELDOUT_TYPES)
def test_e1_mutation_is_not_inert(sweep, mtype):
    eps = episodes_of(sweep, mtype)
    assert eps, mtype
    for e in eps:
        assert not e["agents"]["stale"]["success"], (e["task_id"], e["mutation"])


@pytest.mark.parametrize("mtype", MUTATION_TYPES)
def test_e2_mutation_is_solvable_with_full_reward(sweep, mtype):
    for e in episodes_of(sweep, mtype):
        oracle = e["agents"]["oracle_docs"]
        assert oracle["success"], (e["task_id"], e["mutation"])
        assert oracle["reward"] == 1.0, (e["task_id"], e["mutation"])
        assert oracle["invalid_calls"] == 0
        assert oracle["turns"] == e["min_turns"]


# types whose first stale call does not have to fail: the change shows up in a successful response
SILENT_TYPES = ("none", "pagination_change", "response_field_rename", "enum_value_rename")


def test_error_discovery_is_penalized_but_can_succeed(sweep):
    mutated = [e for e in sweep["episodes"] if e["mutation"]["type"] not in SILENT_TYPES]
    recovered = [e for e in mutated if e["agents"]["oracle_error"]["success"]]
    assert len(recovered) >= 0.8 * len(mutated)
    for e in recovered:
        assert e["agents"]["oracle_error"]["invalid_calls"] >= 1
        assert e["agents"]["oracle_error"]["reward"] < 1.0


def test_e3_docs_reflect_live_schema(sweep):
    for e in sweep["episodes"]:
        if e["mutation"]["type"] != "none":
            assert e["docs_reflect_live"], (e["task_id"], e["mutation"])


def test_e5_e7_e8_adversarial_agents_get_nothing_and_change_nothing(sweep):
    for e in sweep["episodes"]:
        for name in ("claim_success", "malformed_args", "unknown_tool", "docs_only"):
            res = e["agents"][name]
            assert res["reward"] == 0.0, (name, e["task_id"])
            assert res["state_unchanged"], (name, e["task_id"])
        assert e["agents"]["malformed_args"]["invalid_calls"] >= 1
        assert e["agents"]["unknown_tool"]["invalid_calls"] >= 1
        assert e["agents"]["docs_only"]["invalid_calls"] == 0


def test_e6_heldout_types_only_in_test(sweep):
    for e in sweep["episodes"]:
        if e["mutation"]["type"] in HELDOUT_TYPES:
            assert e["split"] == "test", e["task_id"]


def test_coverage_every_type_every_domain(sweep):
    cov = sweep["coverage"]
    for domain in ("calendar", "payments", "ecommerce"):
        for mtype in SEEN_EVAL_TYPES:
            assert cov["train"][domain][mtype] >= 20, (domain, mtype)
        for mtype in TEST_TYPES:
            assert cov["test"][domain][mtype] >= 5, (domain, mtype)
    for mtype in DIVERSITY_TYPES:  # extra training variety; not every type fits every domain
        assert sum(cov["train"][d][mtype] for d in cov["train"]) >= 20, mtype


def test_test_split_is_frozen(sweep):
    """New change types are training-only: the benchmark and all its baselines stay comparable."""
    frozen = json.loads((Path(__file__).parent / "fixtures" / "test_split_episodes_s0.json").read_text())
    now = sorted(({"task_id": e["task_id"], "mutation": e["mutation"], "min_turns": e["min_turns"]}
                  for e in sweep["episodes"] if e["split"] == "test"),
                 key=lambda r: (r["task_id"], r["mutation"]["type"]))
    assert now == frozen


def test_diversity_types_only_in_train_and_val(sweep):
    for e in sweep["episodes"]:
        if e["mutation"]["type"] in DIVERSITY_TYPES:
            assert e["split"] in ("train", "val"), e["task_id"]
    assert set(TRAIN_TYPES) == set(SEEN_EVAL_TYPES) | set(DIVERSITY_TYPES)
    assert set(MUTATION_TYPES) == {"none"} | set(TRAIN_TYPES) | set(HELDOUT_TYPES)


def test_turn_budget_leaves_room_to_adapt(sweep):
    for e in sweep["episodes"]:
        assert e["min_turns"] <= 7, (e["task_id"], e["mutation"])
