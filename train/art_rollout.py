"""ART rollouts: APIShift's own episode loop with ART's vLLM client as the policy.

Each model call's raw Choice (with vLLM token ids and logprobs) is kept, so ART
trains on exactly the tokens the model sampled. The history sent back to the
model never contains earlier reasoning; this matches the eval harness
(harness/policy.py). ART starts a new training sequence whenever a prompt
does not extend the previous one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from apishift.agent.types import AssistantTurn, Message, ToolCall
from apishift.envs.mutations import Spec
from apishift.episode import arun_episode
from apishift.tasks.model import Task

UNPARSEABLE = "\x00unparseable-arguments"
MALFORMED_TOOL = "malformed_tool_call"
_THINK = re.compile(r"^\s*(?:<think>)?(.*?)</think>\s*", re.S)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 3072


def split_reasoning(text: str, reasoning: str = "") -> tuple[str, str]:
    """Return (content, reasoning) with any leading <think>...</think> moved into reasoning."""
    m = _THINK.match(text or "")
    if m and "</think>" in (text or ""):
        return text[m.end():].strip(), (reasoning or m.group(1)).strip()
    return (text or "").strip(), (reasoning or "").strip()


def api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Episode transcript -> OpenAI request messages (drop reasoning and tool names)."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m["content"]}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class ArtPolicy:
    """AsyncPolicy over an OpenAI-compatible client that records every raw Choice."""

    def __init__(self, client: Any, model_name: str, sampling: Sampling, name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.sampling = sampling
        self.name = name
        self.choices: list[Any] = []

    async def aact(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        s = self.sampling
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages(messages),
            tools=tools,
            tool_choice="auto",
            temperature=s.temperature,
            top_p=s.top_p,
            max_tokens=s.max_tokens,
            logprobs=True,
            extra_body={"top_k": s.top_k, "chat_template_kwargs": {"enable_thinking": True}},
        )
        choice = resp.choices[0]
        self.choices.append(choice)
        msg = choice.message
        extra = getattr(msg, "model_extra", None) or {}
        content, reasoning = split_reasoning(msg.content or "",
                                             extra.get("reasoning_content") or extra.get("reasoning") or "")
        calls = tuple(
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "")
            for tc in (msg.tool_calls or [])
        )
        if not calls and "<tool_call>" in content:
            calls = (ToolCall(id=f"malformed_{len(messages)}", name=MALFORMED_TOOL, arguments=UNPARSEABLE),)
        return AssistantTurn(content=content, tool_calls=calls, reasoning=reasoning)


def interleave(messages: list[Message], choices: list[Any]) -> list[Any]:
    """Transcript with each assistant message replaced by the Choice that produced it."""
    it = iter(choices)
    out: list[Any] = []
    for m in api_messages(messages):
        out.append(next(it) if m["role"] == "assistant" else m)
    leftover = next(it, None)
    if leftover is not None:
        raise ValueError("more choices than assistant messages")
    return out


async def rollout(model: Any, task: Task, spec: Spec, seed: int, sampling: Sampling,
                  policy_name: str) -> tuple[Any, dict[str, Any]]:
    """One episode -> (art.Trajectory, APIShift trace)."""
    import art

    policy = ArtPolicy(model.openai_client(), model.get_inference_name(), sampling, policy_name)
    trace = await arun_episode(task, spec, policy, seed)
    traj = art.Trajectory(
        messages_and_choices=interleave(trace["messages"], policy.choices),
        tools=trace["tools"],
        reward=trace["reward"]["total"],
        metrics={"success": trace["success"], "turns": trace["turns"], "invalid_calls": trace["invalid_calls"]},
        metadata={"task_id": task.id, "mutation": spec.type, "seed": seed},
    )
    return traj, trace


def sft_turn_examples(trace: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Successful trace -> one message list per assistant turn (history as the model saw it).

    The last message carries that turn's reasoning inline so the Qwen3 template
    renders it inside <think>; train with assistant_turns="last".
    """
    msgs = trace["messages"]
    examples = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        target: dict[str, Any] = {"role": "assistant",
                                  "content": f"<think>\n{m.get('reasoning_content', '')}\n</think>\n\n{m['content']}"}
        if m.get("tool_calls"):
            target["tool_calls"] = m["tool_calls"]
        examples.append([*api_messages(msgs[:i]), target])
    return examples


def trace_line(trace: dict[str, Any]) -> str:
    return json.dumps(trace, sort_keys=True, ensure_ascii=False)
