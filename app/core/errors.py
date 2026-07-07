"""Turn raw provider/SDK exceptions into short, user-friendly messages."""

# Substrings that indicate an API quota / credit / rate-limit problem across
# Gemini and Groq (their SDKs surface these differently).
QUOTA_MARKERS = [
    "429",
    "resource_exhausted",
    "quota",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "insufficient",
    "credit",
    "billing",
    "exceeded",
    "too many requests",
    "out of capacity",
    "overloaded",
]


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


# Substrings indicating a transient failure that is worth retrying with backoff
# (server-side hiccups, timeouts, and short-lived rate limits).
TRANSIENT_MARKERS = [
    "429",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "resource_exhausted",
    "too many requests",
    "overloaded",
    "unavailable",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "try again",
    "deadline",
]


def is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def friendly_error_message(exc: Exception, context: str = "general") -> str:
    """Map an exception to a message safe to show the end user.

    context: "chat" | "upload" | "ocr" | "general"
    """
    if is_quota_error(exc):
        return (
            "The AI service is currently out of credits or has reached its usage "
            "limit. Sorry for the inconvenience — please try again later."
        )

    if context == "ocr":
        return (
            "We couldn't read the images in this document. Scanned or image-heavy "
            "files can be hard to process — please try a clearer or text-based document."
        )

    if context == "upload":
        return (
            "Something went wrong while processing your document. Please try again "
            "with a different file."
        )

    return "Something went wrong while generating a response. Please try again in a moment."
