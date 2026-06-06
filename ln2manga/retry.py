"""Retry policy for transient OpenAI errors.

The SDK already retries transport-level 429/5xx; this adds a budget-aware application-level
retry around our call sites. We NEVER retry client errors (bad request / auth / permission).
"""
from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

try:  # pragma: no cover - import shape varies slightly by SDK version
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    RETRYABLE: tuple = (RateLimitError, APITimeoutError, APIConnectionError,
                        InternalServerError)
except Exception:  # pragma: no cover
    RETRYABLE = (TimeoutError, ConnectionError)


def with_retry(fn):
    """Decorator: exponential backoff on retryable OpenAI errors, max 6 attempts."""
    return retry(
        reraise=True,
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(6),
    )(fn)
