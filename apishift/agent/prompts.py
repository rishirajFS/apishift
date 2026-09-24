"""Prompts shown to the agent. Deliberately neutral: they never say the tools are stale."""

SYSTEM_PROMPT = (
    "You are an operations assistant with access to the {domain} API through tools. "
    "Complete the user's request by calling tools. Tool results are JSON with an HTTP-style "
    "status. If you are unsure how an endpoint works, call get_api_docs. When the request is "
    "complete, or you cannot make progress, reply with a short final answer and no tool calls."
)


def system_prompt(domain: str) -> str:
    return SYSTEM_PROMPT.format(domain=domain)
