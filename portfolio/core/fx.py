"""
Live currency conversion via yfinance FX tickers (e.g. "EURUSD=X"), used to
blend multi-currency amounts into a single sortable USD number for display.

This is a deliberate, opt-in exception to this app's usual rule of keeping
money unconverted per-currency (see positions.py, pnl.py): Cost Basis and
P&L reflect actual trades and stay native-currency, but a "total dividends
received" figure is a summary stat, not a trade, so converting it is fine -
and it's the only way to make that column sortable at all.
"""

import logging
from functools import lru_cache
from typing import Dict, Optional

import yfinance as yf

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_rate_to_usd(currency: str) -> Optional[float]:
    """
    Today's approximate exchange rate to convert 1 unit of `currency` into
    USD, or None if it couldn't be fetched. Cached per process - this is
    for a display aggregate, not a trade, so exact intraday precision (or
    a fresh fetch on every rerun) doesn't matter.
    """
    currency = (currency or "").upper()
    if currency == "USD":
        return 1.0

    for symbol, invert in ((f"{currency}USD=X", False), (f"USD{currency}=X", True)):
        try:
            hist = yf.Ticker(symbol).history(period="5d")
        except Exception as e:
            logger.warning(f"FX fetch failed for {symbol}: {e}")
            continue
        closes = hist["Close"].dropna() if not hist.empty else hist
        if len(closes) == 0:
            continue
        price = float(closes.iloc[-1])
        return (1.0 / price) if invert else price

    logger.warning(f"No FX rate found for {currency} -> USD")
    return None


def convert_amounts_to_usd(amounts_by_currency: Dict[str, float]) -> Optional[float]:
    """
    Sum a {currency: amount} dict into one USD total. Returns None (rather
    than a partial sum) if any currency's rate couldn't be fetched, so the
    caller can fall back to showing the unconverted breakdown instead of
    silently dropping money that just happened to be in a currency we
    couldn't price.
    """
    if not amounts_by_currency:
        return 0.0
    total = 0.0
    for currency, amount in amounts_by_currency.items():
        rate = get_rate_to_usd(currency)
        if rate is None:
            return None
        total += amount * rate
    return total
