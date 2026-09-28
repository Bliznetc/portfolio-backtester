"""
Shared AI text-generation access - one place for the secrets lookup, the
provider SDK, and guarded no-op behavior every AI-assisted feature in this
app uses (instrument classification, price-dependency suggestions, ...).

Backed by Google's Gemini API (the `google-genai` package). Callers just
call generate_text(prompt) and get a raw string back (or None) - none of
them need to know which provider is behind it.
"""

import logging
from typing import Optional

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"


def _get_api_key() -> Optional[str]:
    if st is None:
        return None
    try:
        return st.secrets["gemini"]["api_key"]
    except Exception:
        return None


def generate_text(prompt: str, max_output_tokens: int = 800) -> Optional[str]:
    """
    Returns the model's raw text response, or None if AI features aren't
    available right now: no key configured (the normal state until secrets
    has a [gemini] api_key), the `google-genai` package isn't installed, or
    the call itself fails for any reason. Callers should treat None as
    "not classified/suggested yet", not an error.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning(
            "google-genai package not installed; run `pip install google-genai` "
            "to enable AI-assisted features."
        )
        return None

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
        )
        return response.text
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        return None


def strip_json_fences(text: str) -> str:
    """Model responses sometimes wrap JSON in ```/```json fences despite being
    asked not to - strip them so json.loads doesn't choke on it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()
