"""Mutation engine: spec types and how each one reshapes the live API.

A spec is a frozen, serializable description of one API change. Each spec is
also the adapter between the live (mutated) surface and the canonical
handlers:

    live_endpoints   canonical schemas -> live schemas (what docs show)
    gone             endpoints that now return 410 with a migration notice
    to_canonical     live request -> canonical request
    render           canonical response body -> live response body
    render_error     default error envelope -> live error envelope

The oracle_* methods do the inverse mapping. Only the scripted oracle agent
uses them; they exist to show that every mutated episode can be solved.

Seeded sampling of specs for a task lives in apishift/sampling.py.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, ClassVar

from apishift.envs.errors import ApiError
from apishift.envs.formats import FORMAT_CHANGES
from apishift.envs.schema import Endpoint, Param

# Seen types appear in train, val and test. Diversity types add training variety only, so the
# test split (and every baseline measured on it) is unchanged. Held-out types are test-only.
SEEN_EVAL_TYPES = ("rename_param", "format_change", "new_required_field", "deprecation_with_migration")
DIVERSITY_TYPES = ("response_field_rename", "enum_value_rename", "type_change", "required_version_param",
                   "compound_change")
TRAIN_TYPES = (*SEEN_EVAL_TYPES, *DIVERSITY_TYPES)
HELDOUT_TYPES = ("pagination_change", "error_schema_change")
TEST_TYPES = ("none", *SEEN_EVAL_TYPES, *HELDOUT_TYPES)
MUTATION_TYPES = ("none", *TRAIN_TYPES, *HELDOUT_TYPES)

DEFAULT_ERROR_FORMAT = (
    "Errors return {status, error: {type, code, message, param?, hint?}}."
)
PROBLEM_ERROR_FORMAT = (
    "Errors follow RFC 7807 (application/problem+json): "
    "{status, type, title, detail, invalid_params?: [{name, reason}]}. "
    "Validation errors use status 422."
)

Args = dict[str, Any]


class _Spec:
    type: ClassVar[str] = "none"
    needs_discovery: ClassVar[bool] = True

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **asdict(self)}  # type: ignore[call-overload]

    def live_endpoints(self, eps: Mapping[str, Endpoint]) -> dict[str, Endpoint]:
        return dict(eps)

    def gone(self) -> dict[str, dict[str, Any]]:
        return {}

    def to_canonical(self, name: str, args: Args) -> tuple[str, Args]:
        return name, args

    def render(self, name: str, live_args: Args, body: dict) -> dict:
        return body

    def render_error(self, envelope: dict) -> dict:
        return envelope

    def error_format(self) -> str:
        return DEFAULT_ERROR_FORMAT

    # oracle side
    def affects(self, name: str, args: Args) -> bool:
        return False

    def oracle_request(self, name: str, args: Args) -> tuple[str, Args]:
        return name, args

    def oracle_response(self, body: dict) -> dict:
        return body

    def affected_endpoints(self, eps: Mapping[str, Endpoint]) -> tuple[str, ...]:
        return ()

    def discovery_keys(self, name: str, args: Args) -> frozenset[str]:
        """What a docs-first agent must still learn before this call (one docs read covers the call)."""
        return frozenset({self.type}) if self.needs_discovery and self.affects(name, args) else frozenset()

    def gone_docs_for(self, name: str) -> dict[str, Any] | None:
        return None


@dataclass(frozen=True)
class NoMutation(_Spec):
    type: ClassVar[str] = "none"
    needs_discovery: ClassVar[bool] = False


@dataclass(frozen=True)
class RenameParam(_Spec):
    type: ClassVar[str] = "rename_param"
    endpoint: str
    old: str
    new: str

    def live_endpoints(self, eps):
        ep = eps[self.endpoint]
        params = tuple(replace(p, name=self.new) if p.name == self.old else p for p in ep.params)
        return {**eps, self.endpoint: ep.with_params(params)}

    def to_canonical(self, name, args):
        if name != self.endpoint or self.new not in args:
            return name, args
        return name, _rename(args, self.new, self.old)

    def affects(self, name, args):
        return name == self.endpoint and self.old in args

    def oracle_request(self, name, args):
        if name != self.endpoint or self.old not in args:
            return name, args
        return name, _rename(args, self.old, self.new)

    def affected_endpoints(self, eps):
        return (self.endpoint,)


@dataclass(frozen=True)
class FormatChangeSpec(_Spec):
    type: ClassVar[str] = "format_change"
    fields: tuple[str, ...]
    change: str

    @property
    def _fc(self):
        return FORMAT_CHANGES[self.change]

    def _touches(self, p: Param) -> bool:
        nested = set((p.items or {}).get("properties", {}))
        return p.name in self.fields or bool(nested & set(self.fields))

    def live_endpoints(self, eps):
        fc = self._fc

        def conv(p: Param) -> Param:
            if p.name not in self.fields:
                return p
            enum = tuple(fc.from_canonical(e) for e in p.enum) if p.enum else None
            return replace(p, type=fc.live_type, fmt=fc.live, enum=enum, minimum=None, maximum=None,
                           items=fc.live_items)

        out = {}
        for n, ep in eps.items():
            if any(self._touches(p) for p in ep.params):
                ep = replace(ep, params=tuple(conv(p) for p in ep.params), notes=(*ep.notes, fc.note))
            out[n] = ep
        return out

    def to_canonical(self, name, args):
        return name, _walk(args, self.fields, self._fc.to_canonical)

    def render(self, name, live_args, body):
        if self._fc.render is not None:
            return self._fc.render(body)
        return _walk(body, self.fields, self._fc.from_canonical)

    def affects(self, name, args):
        return _has_key(args, self.fields)

    def oracle_request(self, name, args):
        return name, _walk(args, self.fields, self._fc.from_canonical)

    def oracle_response(self, body):
        if self._fc.unrender is not None:
            return self._fc.unrender(body)
        return _walk(body, self.fields, self._fc.to_canonical)

    def affected_endpoints(self, eps):
        return tuple(n for n, ep in eps.items() if any(self._touches(p) for p in ep.params))


@dataclass(frozen=True)
class NewRequiredField(_Spec):
    type: ClassVar[str] = "new_required_field"
    endpoint: str
    param: Param
    oracle_value: Any

    def live_endpoints(self, eps):
        ep = eps[self.endpoint]
        return {**eps, self.endpoint: ep.with_params((*ep.params, replace(self.param, required=True)))}

    def to_canonical(self, name, args):
        if name != self.endpoint:
            return name, args
        return name, {k: v for k, v in args.items() if k != self.param.name}

    def affects(self, name, args):
        return name == self.endpoint

    def oracle_request(self, name, args):
        if name != self.endpoint:
            return name, args
        return name, {**args, self.param.name: self.oracle_value}

    def affected_endpoints(self, eps):
        return (self.endpoint,)


@dataclass(frozen=True)
class FixedParam:
    name: str
    type: str
    value: Any
    description: str


@dataclass(frozen=True)
class Deprecation(_Spec):
    type: ClassVar[str] = "deprecation_with_migration"
    old_endpoint: str
    new_endpoint: str
    new_description: str
    param_map: tuple[tuple[str, str], ...] = ()
    fixed: tuple[FixedParam, ...] = ()
    removed_on: str = "2026-09-01"

    def _map(self) -> dict[str, str]:
        return dict(self.param_map)

    def _migration(self) -> dict[str, Any]:
        return {
            "replacement": self.new_endpoint,
            "renamed_parameters": self._map(),
            "added_parameters": {f.name: f.value for f in self.fixed},
        }

    def live_endpoints(self, eps):
        old = eps[self.old_endpoint]
        mapping = self._map()
        params = tuple(replace(p, name=mapping.get(p.name, p.name)) for p in old.params)
        fixed = tuple(
            Param(f.name, f.type, f.description, required=True, enum=(f.value,)) for f in self.fixed
        )
        new = replace(old, name=self.new_endpoint, description=self.new_description,
                      params=params + fixed)
        rest = {n: ep for n, ep in eps.items() if n != self.old_endpoint}
        return {**rest, self.new_endpoint: new}

    def gone(self):
        changes = [f"{a} -> {b}" for a, b in self.param_map]
        changes += [f"add {f.name}={json.dumps(f.value)}" for f in self.fixed]
        hint = f"See get_api_docs('{self.new_endpoint}')."
        if changes:
            hint = "Parameter changes: " + "; ".join(changes) + ". " + hint
        return {
            self.old_endpoint: {
                "message": f"{self.old_endpoint} was removed on {self.removed_on}. "
                f"Use {self.new_endpoint} instead.",
                "hint": hint,
                "migration": self._migration(),
            }
        }

    def gone_docs_for(self, name: str) -> dict[str, Any] | None:
        return self.gone_docs() if name == self.old_endpoint else None

    def gone_docs(self) -> dict[str, Any]:
        return {
            "endpoint": self.old_endpoint,
            "deprecated": True,
            "removed_on": self.removed_on,
            "description": f"REMOVED. Use {self.new_endpoint} instead.",
            "migration": self._migration(),
        }

    def to_canonical(self, name, args):
        if name != self.new_endpoint:
            return name, args
        inverse = {b: a for a, b in self.param_map}
        fixed = {f.name for f in self.fixed}
        return self.old_endpoint, {inverse.get(k, k): v for k, v in args.items() if k not in fixed}

    def affects(self, name, args):
        return name == self.old_endpoint

    def oracle_request(self, name, args):
        if name != self.old_endpoint:
            return name, args
        mapping = self._map()
        mapped = {mapping.get(k, k): v for k, v in args.items()}
        return self.new_endpoint, {**mapped, **{f.name: f.value for f in self.fixed}}

    def affected_endpoints(self, eps):
        return (self.old_endpoint,)


@dataclass(frozen=True)
class PaginationChange(_Spec):
    type: ClassVar[str] = "pagination_change"
    needs_discovery: ClassVar[bool] = False
    endpoint: str
    page_size: int

    def live_endpoints(self, eps):
        ep = eps[self.endpoint]
        extra = (
            Param("limit", "integer", f"Page size (default {self.page_size}).",
                  minimum=1, maximum=self.page_size),
            Param("cursor", "string", "Opaque cursor from a previous response's next_cursor."),
        )
        new = replace(
            ep,
            params=ep.params + extra,
            returns="Object {data: [...], has_more: boolean, next_cursor: string|null}. "
            "Results are paginated; pass next_cursor as cursor to fetch the next page.",
        )
        return {**eps, self.endpoint: new}

    def to_canonical(self, name, args):
        if name != self.endpoint:
            return name, args
        if "cursor" in args:
            _decode_cursor(args["cursor"])
        return name, {k: v for k, v in args.items() if k not in ("limit", "cursor")}

    def render(self, name, live_args, body):
        if name != self.endpoint:
            return body
        items = body["items"]
        offset = _decode_cursor(live_args["cursor"]) if live_args.get("cursor") else 0
        limit = live_args.get("limit") or self.page_size
        end = offset + limit
        has_more = end < len(items)
        return {
            "data": items[offset:end],
            "has_more": has_more,
            "next_cursor": _encode_cursor(end) if has_more else None,
        }

    def affects(self, name, args):
        return name == self.endpoint

    def affected_endpoints(self, eps):
        return (self.endpoint,)


@dataclass(frozen=True)
class ErrorSchemaChange(_Spec):
    """Error envelope moves to RFC 7807 and hints are dropped.

    The inner change is what actually breaks the stale call. It is one of the
    train types, so the novelty the agent has to generalize to is reading the
    new error shape.
    """

    type: ClassVar[str] = "error_schema_change"
    inner: RenameParam | NewRequiredField

    def to_dict(self):
        return {"type": self.type, "inner": self.inner.to_dict()}

    def live_endpoints(self, eps):
        return self.inner.live_endpoints(eps)

    def to_canonical(self, name, args):
        return self.inner.to_canonical(name, args)

    def render_error(self, envelope):
        err = envelope.get("error", {})
        status = 422 if envelope["status"] == 400 else envelope["status"]
        out: dict[str, Any] = {
            "status": status,
            "type": f"https://errors.api.example/{err.get('code', 'unknown')}",
            "title": _TITLES.get(status, "Error"),
            "detail": err.get("message", ""),
        }
        if "param" in err:
            out["invalid_params"] = [{"name": err["param"], "reason": err.get("code", "invalid")}]
        return out

    def error_format(self):
        return PROBLEM_ERROR_FORMAT

    def affects(self, name, args):
        return self.inner.affects(name, args)

    def oracle_request(self, name, args):
        return self.inner.oracle_request(name, args)

    def affected_endpoints(self, eps):
        return self.inner.affected_endpoints(eps)


@dataclass(frozen=True)
class ResponseFieldRename(_Spec):
    """A field is renamed in every response body (requests unchanged), e.g. `email` -> `email_address`.

    Visible in any successful response, so no docs read is needed.
    """

    type: ClassVar[str] = "response_field_rename"
    needs_discovery: ClassVar[bool] = False
    old: str
    new: str

    def live_endpoints(self, eps):
        note = f"Response field '{self.old}' was renamed to '{self.new}'."
        return {n: replace(ep, returns=_swap_word(ep.returns, self.old, self.new), notes=(*ep.notes, note))
                if _mentions(ep.returns, self.old) else ep for n, ep in eps.items()}

    def render(self, name, live_args, body):
        return _rename_keys(body, self.old, self.new)

    def oracle_response(self, body):
        return _rename_keys(body, self.new, self.old)

    def affected_endpoints(self, eps):
        return tuple(n for n, ep in eps.items() if _mentions(ep.returns, self.old))


@dataclass(frozen=True)
class EnumValueRename(_Spec):
    """Values of an enum field are renamed in requests and responses, e.g. order status pending -> open."""

    type: ClassVar[str] = "enum_value_rename"
    field: str
    mapping: tuple[tuple[str, str], ...]
    endpoints: tuple[str, ...]  # endpoints whose request or response carries the field

    def _fwd(self) -> dict[str, str]:
        return dict(self.mapping)

    def _inv(self) -> dict[str, str]:
        return {b: a for a, b in self.mapping}

    def _note(self) -> str:
        pairs = ", ".join(f"'{a}' is now '{b}'" for a, b in self.mapping)
        return f"Values of '{self.field}' changed: {pairs}."

    def live_endpoints(self, eps):
        fwd = self._fwd()

        def conv(p: Param) -> Param:
            if p.name != self.field or not p.enum:
                return p
            return replace(p, enum=tuple(fwd.get(v, v) for v in p.enum))

        return {n: (replace(ep, params=tuple(conv(p) for p in ep.params), notes=(*ep.notes, self._note()))
                    if n in self.endpoints else ep) for n, ep in eps.items()}

    def to_canonical(self, name, args):
        inv = self._inv()
        return name, {k: inv.get(v, v) if k == self.field else v for k, v in args.items()}

    def render(self, name, live_args, body):
        return _map_values(body, self.field, self._fwd())

    def affects(self, name, args):
        return name in self.endpoints

    def oracle_request(self, name, args):
        fwd = self._fwd()
        return name, {k: fwd.get(v, v) if k == self.field else v for k, v in args.items()}

    def oracle_response(self, body):
        return _map_values(body, self.field, self._inv())

    def affected_endpoints(self, eps):
        return tuple(n for n in eps if n in self.endpoints)


@dataclass(frozen=True)
class TypeChangeSpec(FormatChangeSpec):
    """A parameter's JSON type changes, e.g. amounts become decimal strings ("160.99")."""

    type: ClassVar[str] = "type_change"


@dataclass(frozen=True)
class GlobalRequiredParam(_Spec):
    """Every endpoint now requires a version parameter (like a Stripe-Version header)."""

    type: ClassVar[str] = "required_version_param"
    param: Param
    oracle_value: Any

    def live_endpoints(self, eps):
        p = replace(self.param, required=True)
        return {n: ep.with_params((*ep.params, p)) for n, ep in eps.items()}

    def to_canonical(self, name, args):
        return name, {k: v for k, v in args.items() if k != self.param.name}

    def affects(self, name, args):
        return True

    def oracle_request(self, name, args):
        return name, {**args, self.param.name: self.oracle_value}

    def affected_endpoints(self, eps):
        return tuple(eps)


@dataclass(frozen=True)
class Compound(_Spec):
    """Two changes at once. The live API is second(first(canonical))."""

    type: ClassVar[str] = "compound_change"
    first: Any
    second: Any

    def to_dict(self):
        return {"type": self.type, "first": self.first.to_dict(), "second": self.second.to_dict()}

    @property
    def needs_discovery(self) -> bool:  # type: ignore[override]
        return self.first.needs_discovery or self.second.needs_discovery

    def live_endpoints(self, eps):
        return self.second.live_endpoints(self.first.live_endpoints(eps))

    def gone(self):
        return {**self.first.gone(), **self.second.gone()}

    def gone_docs_for(self, name):
        return self.second.gone_docs_for(name) or self.first.gone_docs_for(name)

    def to_canonical(self, name, args):
        return self.first.to_canonical(*self.second.to_canonical(name, args))

    def render(self, name, live_args, body):
        return self.second.render(name, live_args, self.first.render(name, live_args, body))

    def render_error(self, envelope):
        return self.second.render_error(self.first.render_error(envelope))

    def error_format(self):
        return self.second.error_format()

    def affects(self, name, args):
        return self.first.affects(name, args) or self.second.affects(name, args)

    def discovery_keys(self, name, args):
        return (frozenset(f"first:{k}" for k in self.first.discovery_keys(name, args))
                | frozenset(f"second:{k}" for k in self.second.discovery_keys(name, args)))

    def oracle_request(self, name, args):
        return self.second.oracle_request(*self.first.oracle_request(name, args))

    def oracle_response(self, body):
        return self.first.oracle_response(self.second.oracle_response(body))

    def affected_endpoints(self, eps):
        seen = self.first.affected_endpoints(eps)
        return seen + tuple(n for n in self.second.affected_endpoints(eps) if n not in seen)


# Shared site: every domain can require an API version on every request.
API_VERSION_SITE = GlobalRequiredParam(
    Param("api_version", "string", "API version to use. Required on every request.", enum=("2026-09-01",)),
    "2026-09-01",
)

Spec = (
    NoMutation | RenameParam | FormatChangeSpec | NewRequiredField | Deprecation
    | PaginationChange | ErrorSchemaChange | ResponseFieldRename | EnumValueRename
    | TypeChangeSpec | GlobalRequiredParam | Compound
)

_TITLES = {404: "Not Found", 409: "Conflict", 410: "Gone", 422: "Unprocessable Content"}


def _rename(args: Args, old: str, new: str) -> Args:
    return {(new if k == old else k): v for k, v in args.items()}


def _walk(value: Any, fields: tuple[str, ...], fn) -> Any:
    if isinstance(value, dict):
        return {
            k: (fn(v) if k in fields and v is not None else _walk(v, fields, fn))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_walk(v, fields, fn) for v in value]
    return value


def _rename_keys(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {(new if k == old else k): _rename_keys(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename_keys(v, old, new) for v in value]
    return value


def _map_values(value: Any, field: str, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: (mapping.get(v, v) if k == field and isinstance(v, str) else _map_values(v, field, mapping))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_map_values(v, field, mapping) for v in value]
    return value


def _word_re(word: str):
    import re

    return re.compile(rf"(?<![A-Za-z_]){re.escape(word)}(?![A-Za-z_])")


def _mentions(text: str, word: str) -> bool:
    return bool(_word_re(word).search(text or ""))


def _swap_word(text: str, old: str, new: str) -> str:
    return _word_re(old).sub(new, text or "")


def _has_key(value: Any, fields: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(k in fields or _has_key(v, fields) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_key(v, fields) for v in value)
    return False


def _encode_cursor(offset: int) -> str:
    raw = json.dumps({"o": offset}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: Any) -> int:
    bad = ApiError(400, "parameter_invalid", "Invalid cursor.", param="cursor",
                   hint="Pass the next_cursor value from the previous page unchanged.")
    if not isinstance(cursor, str):
        raise bad
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        offset = json.loads(base64.urlsafe_b64decode(padded))["o"]
    except (binascii.Error, ValueError, KeyError, TypeError):
        raise bad from None
    if not isinstance(offset, int) or offset < 0:
        raise bad
    return offset
