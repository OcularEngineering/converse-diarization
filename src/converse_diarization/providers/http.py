"""Retrying HTTP helpers that never leak credentials into error messages."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable

import httpx

from converse_diarization.env import redact
from converse_diarization.providers.base import ProviderError

RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# A 429 that will not clear by waiting: out of credits, quota exhausted, billing disabled.
_PERMANENT_429_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "quota_exceeded",
    "billing_hard_limit_reached",
    "no credits remaining",
    "exceeded your current quota",
)


def _body_snippet(response: httpx.Response | None, limit: int = 300) -> str:
    if response is None:
        return ""
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive; httpx normally decodes
        return ""
    return " ".join(text.split())[:limit]


def is_permanent_429(response: httpx.Response) -> bool:
    """True when a 429 reports exhausted quota or credits rather than a transient rate limit."""
    if response.status_code != 429:
        return False
    body = _body_snippet(response, limit=2000).lower()
    return any(marker in body for marker in _PERMANENT_429_MARKERS)


def _retry_after(response: httpx.Response | None, *, cap: float) -> float | None:
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, cap)


def send_with_retry(
    send: Callable[[], httpx.Response],
    *,
    attempts: int = 6,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    sleep=time.sleep,
) -> httpx.Response:
    """Call ``send`` until it returns a response that is not worth retrying.

    Transport errors, timeouts and the statuses in :data:`RETRY_STATUSES` are retried with
    exponential backoff and jitter, honoring a numeric ``Retry-After`` header. Any other
    response (including non-retryable 4xx) is returned to the caller.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_error: str = "no attempts made"
    last_status: int | None = None
    for attempt in range(attempts):
        response: httpx.Response | None = None
        try:
            response = send()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_status = None
        else:
            if response.status_code not in RETRY_STATUSES or is_permanent_429(response):
                return response
            last_status = response.status_code
            snippet = _body_snippet(response)
            last_error = f"HTTP {response.status_code}" + (f": {snippet}" if snippet else "")
        if attempt + 1 >= attempts:
            break
        delay = min(base_delay_s * (2**attempt), max_delay_s)
        delay += random.uniform(0.0, min(1.0, delay * 0.25))
        hinted = _retry_after(response, cap=max_delay_s)
        if hinted is not None:
            delay = max(delay, hinted)
        sleep(min(delay, max_delay_s))
    raise ProviderError(
        f"request failed after {attempts} attempt(s): {redact(last_error)}",
        retryable=True,
        status=last_status,
    )


def error_from_response(response: httpx.Response, *, context: str) -> ProviderError:
    """Build a redacted :class:`ProviderError` from an unexpected response."""
    snippet = _body_snippet(response) or "<undecodable body>"
    request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
    message = f"{context}: HTTP {response.status_code}"
    if request_id:
        message += f" (request id {request_id})"
    message += f": {snippet}"
    status = response.status_code
    permanent = is_permanent_429(response)
    if permanent:
        message += " [quota or credits exhausted; not retrying]"
    return ProviderError(redact(message), retryable=status in RETRY_STATUSES and not permanent, status=status)


__all__ = ["RETRY_STATUSES", "error_from_response", "is_permanent_429", "send_with_retry"]
