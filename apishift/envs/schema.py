"""Endpoint and parameter schemas: validation, JSON Schema, tool and docs rendering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from apishift.envs.errors import ApiError
from apishift.envs.formats import FORMATS, validate_format

State = dict[str, Any]
# (state, canonical args, id factory) -> (new state, status, body)
Handler = Callable[[State, Mapping[str, Any], Callable[[str], str]], tuple[State, int, dict]]

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    required: bool = False
    enum: tuple[Any, ...] | None = None
    fmt: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    # JSON Schema for array items (shallow validation)
    items: Mapping[str, Any] | None = None

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type, "description": self.describe()}
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.items is not None:
            schema["items"] = dict(self.items)
        if self.fmt == "address_object":
            schema["properties"] = {k: {"type": "string"} for k in ("line1", "city", "postal_code")}
            schema["required"] = ["line1", "city", "postal_code"]
        return schema

    def describe(self) -> str:
        if self.fmt and self.fmt in FORMATS:
            f = FORMATS[self.fmt]
            return f"{self.description} Format: {f.description}."
        return self.description


@dataclass(frozen=True)
class Endpoint:
    name: str
    description: str
    params: tuple[Param, ...]
    returns: str
    handler: Handler
    notes: tuple[str, ...] = field(default=())

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def with_params(self, params: tuple[Param, ...]) -> Endpoint:
        return replace(self, params=params)

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {p.name: p.json_schema() for p in self.params},
            "required": [p.name for p in self.params if p.required],
            "additionalProperties": False,
        }

    def openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def docs(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "endpoint": self.name,
            "description": self.description,
            "parameters": self.json_schema(),
            "returns": self.returns,
        }
        if self.notes:
            doc["notes"] = list(self.notes)
        return doc


def _check_items(p: Param, value: list) -> None:
    item_schema = p.items or {}
    item_type = item_schema.get("type")
    for i, item in enumerate(value):
        if item_type and not _TYPE_CHECKS[item_type](item):
            raise ApiError(400, "parameter_invalid_type",
                           f"Invalid type for '{p.name}[{i}]': expected {item_type}.", param=p.name)
        if item_type == "object":
            props = item_schema.get("properties", {})
            for key in item_schema.get("required", []):
                if key not in item:
                    raise ApiError(400, "parameter_missing",
                                   f"Missing required key '{key}' in '{p.name}[{i}]'.", param=p.name)
            for key, val in item.items():
                if key not in props:
                    raise ApiError(400, "parameter_unknown",
                                   f"Unknown key '{key}' in '{p.name}[{i}]'.", param=p.name)
                if not _TYPE_CHECKS[props[key]["type"]](val):
                    raise ApiError(400, "parameter_invalid_type",
                                   f"Invalid type for '{p.name}[{i}].{key}': expected {props[key]['type']}.",
                                   param=p.name)


def _check_param(endpoint: str, p: Param, value: Any) -> None:
    fmt = FORMATS.get(p.fmt) if p.fmt else None
    if not _TYPE_CHECKS[p.type](value):
        hint = f"Expected {fmt.description}, e.g. {fmt.example!r}." if fmt else None
        raise ApiError(400, "parameter_invalid_type",
                       f"Invalid type for '{p.name}': expected {p.type}.", param=p.name, hint=hint)
    if p.enum is not None and value not in p.enum:
        allowed = ", ".join(repr(e) for e in p.enum)
        raise ApiError(400, "parameter_invalid_value",
                       f"Invalid value for '{p.name}': must be one of {allowed}.", param=p.name)
    if p.minimum is not None and value < p.minimum:
        raise ApiError(400, "parameter_invalid_value",
                       f"'{p.name}' must be >= {p.minimum}.", param=p.name)
    if p.maximum is not None and value > p.maximum:
        raise ApiError(400, "parameter_invalid_value",
                       f"'{p.name}' must be <= {p.maximum}.", param=p.name)
    if p.fmt and not validate_format(p.fmt, value):
        assert fmt is not None
        raise ApiError(400, "parameter_invalid_format",
                       f"Invalid format for '{p.name}': expected {fmt.description}.",
                       param=p.name, hint=f"Example: {fmt.example!r}.")
    if p.type == "array":
        _check_items(p, value)


def validate_args(endpoint: Endpoint, args: Mapping[str, Any]) -> None:
    """Raise ApiError on the first problem, in the order a real API would report it."""
    docs_hint = f"See get_api_docs('{endpoint.name}') for the current parameters."
    for name in args:
        if endpoint.param(name) is None:
            raise ApiError(400, "parameter_unknown", f"Received unknown parameter: {name}",
                           param=name, hint=docs_hint)
    for p in endpoint.params:
        if p.required and args.get(p.name) is None:
            raise ApiError(400, "parameter_missing", f"Missing required parameter: {p.name}",
                           param=p.name, hint=docs_hint)
    for p in endpoint.params:
        if p.name in args and args[p.name] is not None:
            _check_param(endpoint.name, p, args[p.name])
