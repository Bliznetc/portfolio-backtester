"""
Instrument classification: yfinance's sector/industry/business-summary as a
free baseline layer, plus an optional AI-refined thematic tag on top for
cases where that generic GICS-style category doesn't capture what actually
moves the stock (e.g. "AI/HPC data center infrastructure" instead of
yfinance's "Information Technology Services").

The AI layer is a no-op until a [gemini] api_key is added to secrets -
classify_with_ai returns None in that state, same as when the call itself
fails, so callers should treat None as "not classified yet", not an error.
Everything still works with just the yfinance layer in the meantime.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from portfolio.core import ai_client

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

_SUMMARY_CHAR_LIMIT = 1500


@dataclass
class YfClassification:
    sector: Optional[str]
    industry: Optional[str]
    business_summary: Optional[str]


def fetch_yf_classification(symbol: str) -> YfClassification:
    """Best-effort - all fields None if yfinance has nothing for this symbol."""
    if yf is None:
        return YfClassification(None, None, None)
    try:
        info = yf.Ticker(symbol).get_info()
    except Exception:
        return YfClassification(None, None, None)
    return YfClassification(
        sector=info.get("sector"),
        industry=info.get("industry"),
        business_summary=info.get("longBusinessSummary"),
    )


def classify_with_ai(
    ticker: str, sector: Optional[str], industry: Optional[str], business_summary: Optional[str],
) -> Optional[Tuple[str, str, str]]:
    """
    Returns (classification, rationale, model) or None - None whenever
    there's nothing to work with yet: no API key configured (the normal
    state until secrets has a [gemini] api_key), the `google-genai`
    package isn't installed, no business summary to classify from, or the
    call itself fails for any reason.
    """
    if not business_summary:
        return None

    prompt = (
        f"Ticker: {ticker}\n"
        f"Yahoo Finance sector: {sector or 'unknown'}\n"
        f"Yahoo Finance industry: {industry or 'unknown'}\n"
        f"Business summary: {business_summary[:_SUMMARY_CHAR_LIMIT]}\n\n"
        "Classify this company into a short, specific thematic category "
        "that explains what actually drives its stock price - more useful "
        "than a generic sector label (e.g. 'AI/HPC data center "
        "infrastructure' rather than 'Information Technology Services'). "
        "Reply with ONLY a JSON object, no markdown fences: "
        '{"classification": "3-6 word category", "rationale": "one sentence"}'
    )

    text = ai_client.generate_text(prompt, max_output_tokens=200)
    if not text:
        return None

    try:
        data = json.loads(ai_client.strip_json_fences(text))
        classification, rationale = data.get("classification"), data.get("rationale")
        if not classification:
            return None
        return classification, rationale, ai_client.DEFAULT_MODEL
    except Exception as e:
        logger.warning(f"AI classification failed for {ticker}: {e}")
        return None
