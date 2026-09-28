"""
AI-suggested "what influences this instrument's price" categories (e.g.
"Energy Costs", "GPU/Semiconductor Supply"), each with a short rationale
and a handful of representative tickers/ETFs.

There's no public API for this kind of dependency graph - no vendor
publishes one for free - so this is model synthesis grounded in the
instrument's own yfinance sector/industry/business summary, not a lookup.
Treat it as a starting point to verify or correct, not verified fact; the
UI also lets a user add their own categories manually, which don't go
through this module at all.

A no-op (empty list) until a [gemini] api_key is configured in secrets
(see core.ai_client) - callers should treat an empty result as "no
suggestions yet", not an error.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from portfolio.core import ai_client

logger = logging.getLogger(__name__)

_SUMMARY_CHAR_LIMIT = 1500


@dataclass
class SuggestedDependency:
    name: str
    description: str
    representative_tickers: List[str] = field(default_factory=list)


def build_prompt(
    ticker: str, sector: Optional[str], industry: Optional[str], business_summary: str,
    existing_categories: Optional[List[str]] = None,
) -> str:
    """
    Exposed separately from generate_dependencies_with_ai so the exact same
    prompt can be copy-pasted into a chat UI for a manual dry run.

    `existing_categories` is the current shared taxonomy's category names -
    passing it steers the model toward reusing one of them by exact name
    instead of inventing a near-duplicate (e.g. "Electricity Prices" vs
    "Power Availability & Electricity Costs" for the same idea). Without
    it, every instrument tends to grow its own slightly-differently-worded
    categories that never actually get reused.
    """
    reuse_note = ""
    if existing_categories:
        names = ", ".join(sorted(existing_categories))
        reuse_note = (
            "\nThese categories already exist in the shared taxonomy: "
            f"{names}. If one of these is a good match for a factor here, "
            "reuse its exact name (verbatim) instead of inventing a "
            "near-duplicate. Only propose a new category name if none of "
            "the existing ones genuinely fit.\n"
        )
    return (
        f"Ticker: {ticker}\n"
        f"Sector: {sector or 'unknown'}\n"
        f"Industry: {industry or 'unknown'}\n"
        f"Business summary: {business_summary[:_SUMMARY_CHAR_LIMIT]}\n"
        f"{reuse_note}\n"
        "List the top 3 external factors that most influence this "
        "company's stock price (e.g. a commodity, a macro variable, a "
        "supply-chain input, a regulatory factor - not generic things like "
        '"market sentiment" or "the economy"). For each, give a short '
        "category name, a one-sentence description of the link, and 1-3 "
        "real, currently trading tickers or ETFs that represent that "
        "factor.\n\n"
        "Reply with ONLY a JSON array, no markdown fences, in this shape: "
        '[{"name": "...", "description": "...", '
        '"representative_tickers": ["..."]}]'
    )


def parse_response(text: str) -> List[SuggestedDependency]:
    """Parses the JSON-array response shape build_prompt() asks for -
    exposed separately so a manually-pasted answer can go through the same
    parsing path as a live API response."""
    data = json.loads(ai_client.strip_json_fences(text))
    return [
        SuggestedDependency(
            name=item["name"],
            description=item.get("description", ""),
            representative_tickers=[t.upper() for t in item.get("representative_tickers", [])],
        )
        for item in data if item.get("name")
    ]


def generate_dependencies_with_ai(
    ticker: str, sector: Optional[str], industry: Optional[str], business_summary: Optional[str],
    existing_categories: Optional[List[str]] = None,
) -> List[SuggestedDependency]:
    if not business_summary:
        return []

    prompt = build_prompt(ticker, sector, industry, business_summary, existing_categories)
    text = ai_client.generate_text(prompt, max_output_tokens=800)
    if not text:
        return []

    try:
        return parse_response(text)
    except Exception as e:
        logger.warning(f"AI dependency generation failed for {ticker}: {e}")
        return []
