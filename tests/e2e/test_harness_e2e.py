"""E2E test of the Inspect harness with a scripted mock model (no GPU, no network).

Runs real episodes through Inspect: dataset -> solver (APIShift loop with an
Inspect model as policy) -> scorer -> results JSON + traces JSONL.
"""

import json

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from harness.results import summarize, write_results
from harness.task import apishift_eval


def test_harness_runs_tool_calls_and_writes_results(tmp_path):
    outputs = [
        ModelOutput.for_tool_call("mockllm/model", "list_contacts", {}),
        ModelOutput.for_tool_call("mockllm/model", "create_event", {"oops": 1}),
        ModelOutput.from_content("mockllm/model", "I have scheduled the meeting."),
    ]
    model = get_model("mockllm/model", custom_outputs=outputs)
    [log] = inspect_eval(apishift_eval(split="test", seed=0, types="none", limit_per_type=1),
                         model=model, log_dir=str(tmp_path / "logs"), display="none")
    summary, traces = summarize(log, "mock")
    assert summary["n_errors"] == 0 and len(traces) == 1

    trace = traces[0]
    assert [c["name"] for c in trace["calls"]] == ["list_contacts", "create_event"]
    assert trace["calls"][0]["response"]["status"] == 200
    assert trace["calls"][1]["response"]["error"]["code"] == "parameter_unknown"
    # claiming success in text earns nothing; one invalid call is recorded
    assert trace["success"] is False and trace["reward"]["total"] == 0.0
    assert trace["invalid_calls"] == 1 and trace["turns"] == 3

    path = write_results(summary, traces, tmp_path / "results", "mock")
    written = json.loads(path.read_text())
    assert written["by_group"]["control"]["n"] == 1
    lines = (tmp_path / "results" / "traces" / "mock.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["episode_id"] == trace["episode_id"]


def test_unparsed_tool_call_text_counts_as_invalid_call(tmp_path):
    outputs = [
        ModelOutput.from_content("mockllm/model", '<tool_call>\n{"name": "list_contacts", "arguments": {\n'),
        ModelOutput.from_content("mockllm/model", "Giving up."),
    ]
    model = get_model("mockllm/model", custom_outputs=outputs)
    [log] = inspect_eval(apishift_eval(split="test", seed=0, types="none", limit_per_type=1),
                         model=model, log_dir=str(tmp_path / "logs"), display="none")
    _, [trace] = summarize(log, "mock")
    assert trace["turns"] == 2 and trace["invalid_calls"] == 1
    assert trace["calls"][0]["response"]["error"]["code"] == "malformed_arguments"
    assert trace["final_answer"] == "Giving up."


def test_reasoning_is_kept_in_trace_but_not_sent_back(tmp_path):
    from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText
    from inspect_ai.model._model_output import ChatCompletionChoice

    thought = ChatMessageAssistant(content=[ContentReasoning(reasoning="I should list contacts first."),
                                            ContentText(text="Done.")])
    outputs = [ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=thought)])]
    model = get_model("mockllm/model", custom_outputs=outputs)
    [log] = inspect_eval(apishift_eval(split="test", seed=0, types="none", limit_per_type=1),
                         model=model, log_dir=str(tmp_path / "logs"), display="none")
    _, [trace] = summarize(log, "mock")
    final = trace["messages"][-1]
    assert final["reasoning_content"] == "I should list contacts first."
    assert final["content"] == "Done."


def test_dataset_groups_and_heldout_only_in_test():
    test_ds = apishift_eval(split="test", seed=0).dataset
    groups = {s.metadata["group"] for s in test_ds}
    assert groups == {"control", "seen", "heldout"}
    train_ds = apishift_eval(split="train", seed=0).dataset
    assert {s.metadata["group"] for s in train_ds} == {"control", "seen"}
