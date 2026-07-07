"""Shared retry policy for transient upstream failures (5xx, timeouts, rate limits)."""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from app.core.errors import is_transient_error


def with_retry(fn):
    """Retry a callable up to 3 times with exponential backoff on transient errors.

    Non-transient errors (e.g. bad input) fail fast; the final failure re-raises
    so callers can surface a friendly message.
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception(is_transient_error),
        reraise=True,
    )(fn)
