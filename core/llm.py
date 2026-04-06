import os

import google.generativeai as genai
from groq import Groq


def _build_gemini_model():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return None

    genai.configure(api_key=gemini_api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def _build_groq_client():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None
    return Groq(api_key=groq_api_key)


gemini_model = _build_gemini_model()
groq_client = _build_groq_client()

def ask_llm(context, question):
    if not context.strip():
        return "Not found in document"

    prompt = f"""
    You are a legal AI assistant for Indian laws.

    Answer ONLY from the given context.
    Be precise and formal.
    If answer is not present, say: "Not found in document".

    Context:
    {context}

    Question:
    {question}
    """

    if groq_client is not None:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a legal AI assistant for Indian laws."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            text = completion.choices[0].message.content if completion.choices else ""
            return text or "Not found in document"
        except Exception:
            pass

    if gemini_model is None:
        return "LLM is not configured. Set GROQ_API_KEY or GEMINI_API_KEY to enable answering."

    try:
        response = gemini_model.generate_content(prompt)
        return response.text or "Not found in document"
    except Exception:
        return "Unable to generate answer at the moment."