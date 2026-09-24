"""LiveEnv: one episode's sandbox, the canonical API behind a mutation spec.

The agent only ever sees the live surface. `stale_tools()` returns the
ORIGINAL schemas the agent is given. `get_api_docs` always describes the live
(mutated) API.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from apishift.envs.domain import Domain
from apishift.envs.errors import ApiError, error_envelope, ok_envelope
from apishift.envs.mutations import Deprecation, Spec
from apishift.envs.schema import State, validate_args

DOCS_TOOL_NAME = "get_api_docs"
DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": DOCS_TOOL_NAME,
        "description": (
            "Get the current documentation for an API endpoint: parameters, formats and "
            "return shape. Call with no arguments to list all endpoints."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "Endpoint name, e.g. 'create_event'."}
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


class LiveEnv:
    def __init__(self, domain: Domain, state: State, spec: Spec, seed: int) -> None:
        self.domain = domain
        self.spec = spec
        self.state: State = copy.deepcopy(state)
        self._canonical = domain.endpoint_map()
        self._live = spec.live_endpoints(self._canonical)
        self._gone = spec.gone()
        self._seed = seed
        self._counter = 0

    # agent-facing surface -------------------------------------------------

    def stale_tools(self) -> list[dict[str, Any]]:
        return [e.openai_tool() for e in self.domain.endpoints] + [DOCS_TOOL]

    def execute(self, name: str, raw_arguments: str) -> dict[str, Any]:
        try:
            args = _parse_args(raw_arguments)
            if name == DOCS_TOOL_NAME:
                return ok_envelope(200, self._docs(args))
            if name in self._gone:
                info = self._gone[name]
                raise ApiError(410, "endpoint_removed", info["message"], hint=info["hint"],
                               extra={"migration": info["migration"]})
            endpoint = self._live.get(name)
            if endpoint is None:
                raise ApiError(404, "unknown_endpoint", f"Unknown endpoint: {name}",
                               hint="Call get_api_docs() to list available endpoints.")
            validate_args(endpoint, args)
            cname, cargs = self.spec.to_canonical(name, args)
            canonical = self._canonical[cname]
            validate_args(canonical, cargs)
            new_state, status, body = canonical.handler(self.state, cargs, self._new_id)
            self.state = new_state
            return ok_envelope(status, self.spec.render(name, args, copy.deepcopy(body)))
        except ApiError as err:
            return self.spec.render_error(error_envelope(err))

    # docs -----------------------------------------------------------------

    def docs_for(self, endpoint: str) -> dict[str, Any]:
        return self._docs({"endpoint": endpoint})

    def stale_docs_for(self, endpoint: str) -> dict[str, Any]:
        return self._canonical[endpoint].docs()

    def _docs(self, args: dict[str, Any]) -> dict[str, Any]:
        unknown = [k for k in args if k != "endpoint"]
        if unknown:
            raise ApiError(400, "parameter_unknown", f"Received unknown parameter: {unknown[0]}",
                           param=unknown[0])
        name = args.get("endpoint")
        if name is None:
            return {
                "api": self.domain.name,
                "description": self.domain.description,
                "endpoints": [{"name": n, "description": e.description} for n, e in self._live.items()],
                "error_format": self.spec.error_format(),
            }
        if not isinstance(name, str):
            raise ApiError(400, "parameter_invalid_type", "Invalid type for 'endpoint': expected string.",
                           param="endpoint")
        if name in self._gone and isinstance(self.spec, Deprecation):
            return self.spec.gone_docs()
        endpoint = self._live.get(name)
        if endpoint is None:
            raise ApiError(404, "unknown_endpoint", f"No documentation for endpoint: {name}",
                           param="endpoint",
                           hint="Call get_api_docs() with no arguments to list endpoints.")
        return {**endpoint.docs(), "error_format": self.spec.error_format()}

    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        digest = hashlib.sha256(f"{self._seed}:{self.domain.name}:{self._counter}".encode()).hexdigest()
        return f"{prefix}_{digest[:10]}"


def _parse_args(raw: str) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise ApiError(400, "malformed_arguments", "Request body is not valid JSON.",
                       hint="Tool arguments must be a JSON object.") from None
    if not isinstance(parsed, dict):
        raise ApiError(400, "malformed_arguments", "Request body must be a JSON object.")
    return parsed
