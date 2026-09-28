"""E2E: full episodes through the ART policy adapter, with a scripted fake model server.

The fake server replays the docs-first oracle, emits Qwen3-style
"<think>...</think>" text and OpenAI tool calls, and returns real openai
Choice objects. It checks what training relies on: reasoning is split out
and never sent back, every assistant message maps to exactly one Choice, the
reward is exact, and SFT examples come one per turn with the reasoning inline.
"""

import asyncio
import json

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function

from apishift.agent.scripted import OracleAgent
from apishift.episode import arun_episode
from apishift.sampling import sample_mutation
from apishift.tasks import all_tasks
from train.art_rollout import ArtPolicy, Sampling, api_messages, interleave, sft_turn_examples


class _Resp:
    def __init__(self, choice):
        self.choices = [choice]


class FakeServer:
    """Replays an oracle; records every request so the test can inspect what was sent."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.requests = []
        self.chat = self
        self.completions = self

    async def create(self, **req):
        self.requests.append(req)
        turn = self.oracle.act(req["messages"], req["tools"])
        calls = [ChatCompletionMessageToolCall(id=tc.id, type="function",
                                               function=Function(name=tc.name, arguments=tc.arguments))
                 for tc in turn.tool_calls] or None
        content = f"<think>\nplanning step {len(self.requests)}\n</think>\n\n{turn.content}"
        msg = ChatCompletionMessage(role="assistant", content=content, tool_calls=calls)
        return _Resp(Choice(index=0, finish_reason="tool_calls" if calls else "stop", message=msg))


def run(task, spec):
    server = FakeServer(OracleAgent(task, spec, "docs"))
    policy = ArtPolicy(server, "fake", Sampling(), "fake")
    trace = asyncio.run(arun_episode(task, spec, policy, 0))
    return server, policy, trace


def test_oracle_through_art_policy_gets_full_reward_and_clean_history():
    task = next(t for t in all_tasks() if t.split == "train" and t.template == "refund_full")
    for mtype in ("none", "rename_param", "deprecation_with_migration"):
        spec = sample_mutation(task, mtype, 0)
        server, policy, trace = run(task, spec)
        assert trace["success"] and trace["reward"]["total"] == 1.0, mtype
        for req in server.requests:
            for m in req["messages"]:
                assert "reasoning_content" not in m and "<think>" not in str(m.get("content")), mtype
            assert req["logprobs"] is True
            assert req["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
        assistants = [m for m in trace["messages"] if m["role"] == "assistant"]
        assert all(m["reasoning_content"].startswith("planning step") for m in assistants)
        mac = interleave(trace["messages"], policy.choices)
        assert sum(isinstance(x, Choice) for x in mac) == len(assistants) == len(policy.choices)
        assert len(mac) == len(api_messages(trace["messages"]))


def test_prompt_variant_reaches_the_model_through_art_policy():
    task = next(t for t in all_tasks() if t.split == "train" and t.template == "refund_full")
    spec = sample_mutation(task, "required_version_param", 0)
    server = FakeServer(OracleAgent(task, spec, "docs"))
    trace = asyncio.run(arun_episode(task, spec, ArtPolicy(server, "fake", Sampling(), "fake"), 0,
                                     prompt_variant="recovery"))
    assert trace["success"] and trace["prompt_variant"] == "recovery"
    assert all("may have changed" in req["messages"][0]["content"] for req in server.requests)


def test_rollout_forwards_prompt_variant():
    import inspect

    from train.art_rollout import rollout

    assert "prompt_variant" in inspect.signature(rollout).parameters


def test_sft_examples_one_per_turn_with_inline_reasoning():
    task = next(t for t in all_tasks() if t.split == "train" and t.template == "cancel_order")
    _, _, trace = run(task, sample_mutation(task, "new_required_field", 0))
    examples = sft_turn_examples(trace)
    assert len(examples) == trace["turns"]
    for ex in examples:
        *history, target = ex
        assert target["role"] == "assistant" and target["content"].startswith("<think>\nplanning step")
        assert all("<think>" not in str(m.get("content")) for m in history)
        assert history[0]["role"] == "system" and history[1]["role"] == "user"
    assert json.loads(examples[0][-1]["tool_calls"][0]["function"]["arguments"]) is not None
