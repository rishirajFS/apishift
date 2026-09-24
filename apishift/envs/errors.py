"""Structured API errors and response envelopes.

The default error envelope follows the common Stripe-like shape:
    {"status": 400, "error": {"type", "code", "message", "param", "hint"}}
"""

from __future__ import annotations

from typing import Any

ERROR_TYPES = {
    400: "invalid_request_error",
    404: "not_found_error",
    409: "conflict_error",
    410: "gone_error",
    422: "validation_error",
}


class ApiError(Exception):
    """Raised by validation and handlers; rendered into an error envelope."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        param: str | None = None,
        hint: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.param = param
        self.hint = hint
        self.extra = dict(extra or {})


def error_envelope(err: ApiError) -> dict[str, Any]:
    error: dict[str, Any] = {
        "type": ERROR_TYPES.get(err.status, "api_error"),
        "code": err.code,
        "message": err.message,
    }
    if err.param is not None:
        error["param"] = err.param
    if err.hint is not None:
        error["hint"] = err.hint
    error.update(err.extra)
    return {"status": err.status, "error": error}


def ok_envelope(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"status": status, "body": body}


def not_found(kind: str, ident: str, param: str) -> ApiError:
    return ApiError(404, "resource_missing", f"No such {kind}: '{ident}'", param=param)


def conflict(message: str, param: str | None = None) -> ApiError:
    return ApiError(409, "conflict", message, param=param)


def invalid(message: str, param: str | None = None, hint: str | None = None) -> ApiError:
    return ApiError(400, "parameter_invalid", message, param=param, hint=hint)
