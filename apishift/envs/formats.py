"""Named value formats and the format changes used by `format_change`.

A Format validates a value and describes itself for docs and error messages.
A FormatChange maps a canonical format to a live one, with converters in both
directions: live -> canonical for requests, canonical -> live for responses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from apishift.envs.errors import ApiError

_DATETIME_LOCAL = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_US = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_ISO_DT = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::\d{2})?(?:Z|[+-]00:00)$")
_ISO_DURATION = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$")
_GID = re.compile(r"^gid://api/([a-z]+_[0-9a-f]+)$")
GID_PREFIX = "gid://api/"


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _two_decimals(v: Any) -> bool:
    return _is_num(v) and v >= 0 and abs(round(v * 100) - v * 100) < 1e-6


def _iso_duration_ok(v: Any) -> bool:
    if not _is_str(v):
        return False
    m = _ISO_DURATION.match(v)
    return bool(m and (m.group(1) or m.group(2)))


_ADDRESS_KEYS = ("line1", "city", "postal_code")


def _address_obj_ok(v: Any) -> bool:
    return (
        isinstance(v, dict)
        and set(v) == set(_ADDRESS_KEYS)
        and all(_is_str(v[k]) and v[k].strip() and "," not in v[k] for k in _ADDRESS_KEYS)
    )


def _attendee_objs_ok(v: Any) -> bool:
    return isinstance(v, list) and all(
        isinstance(a, dict) and set(a) == {"email"} and _is_str(a["email"]) and bool(_EMAIL.match(a["email"]))
        for a in v
    )


def _address_str_ok(v: Any) -> bool:
    return _is_str(v) and len([p for p in v.split(", ") if p.strip()]) == 3


@dataclass(frozen=True)
class Format:
    name: str
    description: str
    example: Any
    validate: Callable[[Any], bool]


FORMATS: dict[str, Format] = {
    f.name: f
    for f in (
        Format(
            "datetime_local",
            "date-time as 'YYYY-MM-DD HH:MM' (24h, UTC)",
            "2026-10-01 14:00",
            lambda v: _is_str(v) and bool(_DATETIME_LOCAL.match(v)),
        ),
        Format("date", "date as 'YYYY-MM-DD'", "2026-10-01", lambda v: _is_str(v) and bool(_DATE.match(v))),
        Format("date_us", "date as 'MM/DD/YYYY'", "10/15/2026", lambda v: _is_str(v) and bool(_DATE_US.match(v))),
        Format("email", "email address", "ana@example.com", lambda v: _is_str(v) and bool(_EMAIL.match(v))),
        Format("money_dollars", "amount in dollars, up to 2 decimals", 25.5, _two_decimals),
        Format(
            "iso8601_datetime",
            "ISO 8601 date-time in UTC, e.g. 'YYYY-MM-DDTHH:MM:SSZ'",
            "2026-10-01T14:00:00Z",
            lambda v: _is_str(v) and bool(_ISO_DT.match(v)),
        ),
        Format(
            "iso8601_date",
            "ISO 8601 date 'YYYY-MM-DD'",
            "2026-10-15",
            lambda v: _is_str(v) and bool(_DATE.match(v)),
        ),
        Format(
            "money_cents",
            "integer amount in the smallest currency unit (cents)",
            2550,
            lambda v: _is_int(v) and v >= 0,
        ),
        Format(
            "iso8601_duration",
            "ISO 8601 duration, e.g. 'PT15M' or 'PT1H30M'",
            "PT15M",
            _iso_duration_ok,
        ),
        Format(
            "gid",
            "global ID of the form 'gid://api/<id>'",
            "gid://api/evt_0a1b2c3d4e",
            lambda v: _is_str(v) and bool(_GID.match(v)),
        ),
        Format(
            "attendee_objects",
            "list of attendee objects, each {\"email\": string}",
            [{"email": "ana@example.com"}],
            _attendee_objs_ok,
        ),
        Format(
            "enum_upper",
            "upper-case enum value",
            "DAMAGED",
            lambda v: _is_str(v) and v == v.upper(),
        ),
        Format(
            "decimal_string",
            "amount as a decimal string with exactly 2 decimals",
            "160.99",
            lambda v: _is_str(v) and bool(re.fullmatch(r"\d+\.\d{2}", v)),
        ),
        Format(
            "int_string",
            "whole number as a string",
            "2",
            lambda v: _is_str(v) and bool(re.fullmatch(r"[1-9]\d*", v)),
        ),
        Format(
            "address_string",
            "single line 'street, city, postal code'",
            "12 Oak St, Springfield, 12345",
            _address_str_ok,
        ),
        Format(
            "address_object",
            "object with keys line1, city, postal_code",
            {"line1": "12 Oak St", "city": "Springfield", "postal_code": "12345"},
            _address_obj_ok,
        ),
    )
}


@dataclass(frozen=True)
class FormatChange:
    name: str
    canonical: str
    live: str
    live_type: str
    to_canonical: Callable[[Any], Any]
    from_canonical: Callable[[Any], Any]
    note: str
    live_items: Mapping[str, Any] | None = None
    # Optional whole-body response transforms; default converts matching keys.
    render: Callable[[Any], Any] | None = None
    unrender: Callable[[Any], Any] | None = None


def _iso_to_local(v: str) -> str:
    m = _ISO_DT.match(v)
    assert m is not None
    return f"{m.group(1)} {m.group(2)}"


def _local_to_iso(v: str) -> str:
    return v.replace(" ", "T") + ":00Z"


def _iso_date_to_us(v: str) -> str:
    y, m, d = v.split("-")
    return f"{m}/{d}/{y}"


def _us_to_iso_date(v: str) -> str:
    m, d, y = v.split("/")
    return f"{y}-{m}-{d}"


def _duration_to_minutes(v: str) -> int:
    m = _ISO_DURATION.match(v)
    assert m is not None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def _minutes_to_duration(v: int) -> str:
    hours, minutes = divmod(v, 60)
    return "PT" + (f"{hours}H" if hours else "") + (f"{minutes}M" if minutes or not hours else "")


def _gid_to_id(v: Any) -> str:
    m = _GID.match(v) if isinstance(v, str) else None
    if m is None:
        raise ApiError(400, "parameter_invalid_format",
                       f"Invalid ID {v!r}: expected a global ID like 'gid://api/<id>'.")
    return m.group(1)


def _add_gids(value: Any) -> Any:
    """Shopify-style: keep legacy `id`, add a `gid` alongside it."""
    if isinstance(value, dict):
        out = {k: _add_gids(v) for k, v in value.items()}
        if isinstance(value.get("id"), str):
            out["gid"] = GID_PREFIX + value["id"]
        return out
    if isinstance(value, list):
        return [_add_gids(v) for v in value]
    return value


def _obj_to_address(v: dict) -> str:
    return ", ".join(v[k].strip() for k in _ADDRESS_KEYS)


def _address_to_obj(v: str) -> dict:
    return dict(zip(_ADDRESS_KEYS, v.split(", "), strict=True))


FORMAT_CHANGES: dict[str, FormatChange] = {
    c.name: c
    for c in (
        FormatChange("datetime_to_iso8601", "datetime_local", "iso8601_datetime", "string",
                     _iso_to_local, _local_to_iso,
                     "Date-times are now ISO 8601 in UTC, e.g. 2026-10-01T14:00:00Z."),
        FormatChange("us_date_to_iso8601", "date_us", "iso8601_date", "string",
                     _iso_date_to_us, _us_to_iso_date, "Dates are now ISO 8601 (YYYY-MM-DD)."),
        FormatChange("dollars_to_cents", "money_dollars", "money_cents", "integer",
                     lambda v: v / 100, lambda v: round(v * 100),
                     "Amounts are now integers in the smallest currency unit (cents)."),
        FormatChange("minutes_to_iso8601_duration", "minutes", "iso8601_duration", "string",
                     _duration_to_minutes, _minutes_to_duration,
                     "Offsets are now ISO 8601 durations, e.g. PT15M."),
        FormatChange("address_string_to_object", "address_string", "address_object", "object",
                     _obj_to_address, _address_to_obj,
                     "Addresses are now objects {line1, city, postal_code}."),
        FormatChange("ids_to_global_ids", "id", "gid", "string",
                     _gid_to_id, lambda v: GID_PREFIX + v,
                     "ID parameters now take global IDs (the `gid` field of each object, "
                     "gid://api/<id>). Legacy `id` values are no longer accepted as input.",
                     render=_add_gids, unrender=lambda body: body),
        FormatChange("attendees_to_objects", "email_list", "attendee_objects", "array",
                     lambda v: [a["email"] for a in v], lambda v: [{"email": e} for e in v],
                     "Attendees are now objects {email}.",
                     live_items={"type": "object", "properties": {"email": {"type": "string"}},
                                 "required": ["email"]}),
        FormatChange("dollars_to_decimal_string", "money_dollars", "decimal_string", "string",
                     lambda v: float(v), lambda v: f"{v:.2f}",
                     "Amounts are now decimal strings, e.g. \"160.99\"."),
        FormatChange("int_to_string", "int", "int_string", "string",
                     lambda v: int(v), lambda v: str(v),
                     "Whole-number parameters are now sent as strings, e.g. \"2\"."),
        FormatChange("enum_to_upper", "enum_lower", "enum_upper", "string",
                     lambda v: v.lower(), lambda v: v.upper(),
                     "Enum values are now UPPER_CASE."),
    )
}


def validate_format(fmt: str, value: Any) -> bool:
    spec = FORMATS.get(fmt)
    return True if spec is None else spec.validate(value)
