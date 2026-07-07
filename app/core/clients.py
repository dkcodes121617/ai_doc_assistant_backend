"""Lazily-created, process-wide singleton clients.

Re-creating SDK clients on every request adds latency and connection churn.
These helpers build each client once and share it across requests.
"""
import threading

from google import genai
from google.genai import types
from groq import Groq

from app.core.config import settings

# Network timeouts so a hung upstream can't hang a request forever.
_GENAI_TIMEOUT_MS = 60_000  # google-genai expects milliseconds
_GROQ_TIMEOUT_S = 60.0

_lock = threading.Lock()
_genai_client: genai.Client | None = None
_groq_client: Groq | None = None


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        with _lock:
            if _genai_client is None:
                _genai_client = genai.Client(
                    api_key=settings.GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=_GENAI_TIMEOUT_MS),
                )
    return _genai_client


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        with _lock:
            if _groq_client is None:
                _groq_client = Groq(
                    api_key=settings.GROQ_API_KEY,
                    timeout=_GROQ_TIMEOUT_S,
                )
    return _groq_client
