import logging

from google.genai import types

from app.core.clients import get_genai_client, get_groq_client

logger = logging.getLogger(__name__)


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """Helper to get a full string response without streaming."""
    result = []
    for chunk in call_llm(system_prompt, user_prompt):
        result.append(chunk)
    return "".join(result)

def call_llm(system_prompt: str, user_prompt: str):
    """Stream from Gemini, falling back to Groq only if Gemini fails before
    emitting anything. Raises on total failure so callers can surface a clean
    message (avoids streaming a partial answer then duplicating it with Groq)."""
    emitted = False

    # --- Primary: Gemini ---
    try:
        client = get_genai_client()
        response = client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        for chunk in response:
            if chunk.text:
                emitted = True
                yield chunk.text
        return  # completed successfully
    except Exception as e:
        # Copy out of the except target (Python unbinds it after the block).
        gemini_error = e
        logger.warning("Gemini failed: %s", gemini_error)
        if emitted:
            # Partial answer already streamed — do not re-run Groq (would duplicate).
            raise RuntimeError(f"LLM stream interrupted: {gemini_error}")

    # --- Fallback: Groq (only reached if Gemini emitted nothing) ---
    try:
        groq_client = get_groq_client()
        stream = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as groq_error:
        logger.error("Groq fallback failed: %s", groq_error)
        # Preserve the original error text so quota/credit markers survive.
        raise RuntimeError(f"All AI providers failed: {gemini_error}; {groq_error}")
