from google import genai
from google.genai import types
from groq import Groq
from app.core.config import settings

def generate_text(system_prompt: str, user_prompt: str) -> str:
    """Helper to get a full string response without streaming."""
    result = []
    for chunk in call_llm(system_prompt, user_prompt):
        result.append(chunk)
    return "".join(result)

def call_llm(system_prompt: str, user_prompt: str):
    use_groq = False
    
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        iterator = iter(response)
        first_chunk = next(iterator)
        if first_chunk.text:
            yield first_chunk.text
        for chunk in iterator:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        print(f"Gemini failed, falling back to Groq: {e}")
        use_groq = True
        
    if use_groq:
        try:
            groq_client = Groq(api_key=settings.GROQ_API_KEY)
            stream = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-8b-8192",
                stream=True
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            print(f"Groq fallback failed: {e}")
            yield f"\n\n[Error: Both primary (Gemini) and fallback (Groq) LLMs failed to respond. Details: {e}]"
