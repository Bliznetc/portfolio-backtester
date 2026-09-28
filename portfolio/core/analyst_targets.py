"""
Analyst 12-month price targets and recommendation consensus, sourced from
yfinance's quote-summary data (the same source Yahoo Finance's website shows
under "Analyst Price Targets"). Coverage is inconsistent - ETFs, indices,
and many non-US or small-cap tickers have no analyst coverage at all, so a
missing entry here is the normal case, not an error.
"""

from dataclasses import dataclass
from typing import List, Optional

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class PriceTargets:
    mean: Optional[float]
    median: Optional[float]
    high: Optional[float]
    low: Optional[float]
    num_analysts: Optional[int]
    recommendation_key: Optional[str]     # e.g. "buy", "hold", "sell"
    recommendation_mean: Optional[float]  # 1.0 (strong buy) - 5.0 (strong sell)


@dataclass
class AnalystRating:
    firm: str
    grade: Optional[str]        # e.g. "Buy", "Overweight", "Neutral"
    price_target: Optional[float]
    date: str


def fetch_price_targets(symbol: str, price_scale: float = 1.0) -> Optional[PriceTargets]:
    """
    None if yfinance has no analyst coverage for this symbol (the normal
    case for ETFs/indices/many non-US tickers) or the lookup fails.

    `price_scale` must be the same GBX/GBP scale
    symbol_resolver.resolve_yfinance_symbol returned for this ticker's
    current price - Yahoo quotes price targets in the same units as the
    regular quote (pence for LSE lines), so the same rescale applies.
    """
    if yf is None:
        return None
    try:
        info = yf.Ticker(symbol).get_info()
    except Exception:
        return None

    mean = info.get("targetMeanPrice")
    if mean is None:
        return None

    def scaled(field):
        value = info.get(field)
        return value * price_scale if value is not None else None

    return PriceTargets(
        mean=scaled("targetMeanPrice"),
        median=scaled("targetMedianPrice"),
        high=scaled("targetHighPrice"),
        low=scaled("targetLowPrice"),
        num_analysts=info.get("numberOfAnalystOpinions"),
        recommendation_key=info.get("recommendationKey"),
        recommendation_mean=info.get("recommendationMean"),
    )


def fetch_analyst_ratings(symbol: str, price_scale: float = 1.0) -> List[AnalystRating]:
    """
    Each covering firm's most recent rating/price target, sourced from
    yfinance's analyst upgrades/downgrades history - that history has one
    row per rating change over time, so this dedupes to each firm's latest
    action and sorts most-recent-first. A firm whose latest action carries
    no numeric target (a grade-only reiteration) reports price_target=None
    rather than Yahoo's 0.0 placeholder. Empty list if yfinance has no
    upgrade/downgrade history for this symbol at all.
    """
    if yf is None:
        return []
    try:
        df = yf.Ticker(symbol).upgrades_downgrades
    except Exception:
        return []
    if df is None or df.empty:
        return []

    df = df.sort_index(ascending=False)
    seen = set()
    ratings = []
    for date, row in df.iterrows():
        firm = row.get("Firm")
        if not firm or firm in seen:
            continue
        seen.add(firm)
        target = row.get("currentPriceTarget")
        ratings.append(AnalystRating(
            firm=firm,
            grade=row.get("ToGrade") or None,
            price_target=(target * price_scale) if target else None,
            date=date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
        ))
    return ratings
