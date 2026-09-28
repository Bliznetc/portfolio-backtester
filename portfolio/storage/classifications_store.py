"""
Storage for instrument classifications: yfinance's sector/industry/business
summary plus an optional AI-refined thematic tag on top. Keyed by the same
yfinance symbol used for price/analyst-target lookups (see
core.symbol_resolver) - a global table, not scoped to a user or portfolio,
since a classification belongs to the instrument itself and every portfolio
holding the same instrument should share one classification, not re-derive
or re-pay for it.
"""

from typing import Optional

from . import db


def get_classification(yf_symbol: str) -> Optional[dict]:
    rows = db.execute(
        "SELECT * FROM instrument_classifications WHERE yf_symbol = %s",
        (yf_symbol,),
        fetch=True,
    )
    return rows[0] if rows else None


def upsert_yf_layer(
    yf_symbol: str, display_ticker: str, sector: Optional[str],
    industry: Optional[str], business_summary: Optional[str],
) -> None:
    db.execute(
        """
        INSERT INTO instrument_classifications
            (yf_symbol, display_ticker, yf_sector, yf_industry, business_summary)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (yf_symbol) DO UPDATE SET
            display_ticker = EXCLUDED.display_ticker,
            yf_sector = EXCLUDED.yf_sector,
            yf_industry = EXCLUDED.yf_industry,
            business_summary = EXCLUDED.business_summary,
            updated_at = now()
        """,
        (yf_symbol, display_ticker, sector, industry, business_summary),
    )


def upsert_ai_layer(yf_symbol: str, ai_classification: str, ai_rationale: str, ai_model: str) -> None:
    db.execute(
        """
        UPDATE instrument_classifications
        SET ai_classification = %s, ai_rationale = %s, ai_model = %s,
            ai_generated_at = now(), updated_at = now()
        WHERE yf_symbol = %s
        """,
        (ai_classification, ai_rationale, ai_model, yf_symbol),
    )
