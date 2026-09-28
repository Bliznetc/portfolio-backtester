"""
Best-effort mapping from a broker's ticker to a yfinance-fetchable symbol.

Trading 212 (and brokers generally) report a plain ticker like "BP" or
"LGEN" with no exchange info attached. yfinance needs an exchange suffix for
anything not listed on a US exchange (e.g. "BP.L" for the London Stock
Exchange). There is no reliable general algorithm for this - the same plain
ticker can exist on several exchanges - so this handles two things:

1. A narrow, data-driven heuristic for the one case we can resolve with real
   confidence: an ISIN starting "GB" combined with a price quoted in GBX
   (pence), which in practice means "a main-market London Stock Exchange
   equity".
2. A manual per-ISIN override table (_ISIN_OVERRIDES below) for individual
   instruments verified by hand - mainly UCITS ETFs, which list across many
   European exchanges (Xetra, Borsa Italiana, Euronext, ...) with no single
   reliable rule for which one a given broker quotes. Add to this table only
   after confirming via `yfinance.Search(isin)` that the symbol's ISIN
   actually matches and its price is in the right ballpark - see NQSE.DE
   below for the pattern (found by searching IE00BYVQ9F29 directly). An
   override is only usable if the currency yfinance quotes it in matches
   the currency our own data already has for that ticker - if the only
   listing we can find is priced in a different currency than the broker
   reported (see CNX1 in the caller's "unresolved" notes), converting that
   would need a live FX rate, not a fixed scale, so it's left unresolved
   rather than silently blended.
3. A general rule for US multi-class tickers (e.g. "BRK.B"): yfinance
   always uses a hyphen instead of a period for the share class ("BRK-B"),
   so any ticker containing "." gets that substitution tried.

Anything not covered by any of these is left as the plain ticker; if that
isn't fetchable, callers should treat it as unpriced rather than guess further.
"""

from typing import Optional, Tuple

# GBX (pence) is the tell: LSE main-market equities are quoted in pence,
# and Trading 212's importer already divides GBX prices by 100 before they
# reach this point, so `price_currency` shows "GBP" here for what was
# originally a GBX quote. A GB-prefixed ISIN alone isn't a safe enough
# signal on its own (some GB-domiciled companies price in USD on other
# exchanges), so both conditions are required together.
_LIKELY_GBX_ORIGIN_CURRENCY = "GBP"
_LSE_ISIN_PREFIX = "GB"

# isin -> (yfinance_symbol, price_scale). Verified by hand against a real
# quote in the currency our own data already has for that ticker (see the
# module docstring) - extend this as new unresolved tickers turn up, not by
# guessing.
_ISIN_OVERRIDES = {
    # iShares NASDAQ 100 UCITS ETF (Acc) - Revolut trades this under "NQSE"
    # with no exchange suffix; Yahoo only has it listed on Xetra. EUR both sides.
    "IE00BYVQ9F29": ("NQSE.DE", 1.0),
    # iShares Physical Gold ETC - Trading 212's "CSGOLD". USD both sides.
    "CH0104136236": ("CSGOLD.SW", 1.0),
    # iShares Physical Silver ETC - Trading 212's "ISLN". USD both sides.
    "IE00B4NCWG09": ("ISLN.L", 1.0),
    # Samsung Electronics GDR (London IOB) - Trading 212's "SMSN". USD both sides.
    "US7960508882": ("SMSN.IL", 1.0),
}


def resolve_yfinance_symbol(
    ticker: str, isin: Optional[str], price_currency: Optional[str]
) -> Tuple[str, float]:
    """
    Returns (yfinance_symbol, price_scale).

    `price_scale` must be multiplied into whatever price yfinance returns
    for that symbol before comparing it against our own data: yfinance
    quotes LSE lines in pence (GBX) even though our stored price_currency
    for the same instrument is normalized to GBP, so a ".L" symbol here
    always pairs with a 0.01 scale to bring it back to GBP.
    """
    if isin and isin in _ISIN_OVERRIDES:
        return _ISIN_OVERRIDES[isin]
    if isin and isin.startswith(_LSE_ISIN_PREFIX) and price_currency == _LIKELY_GBX_ORIGIN_CURRENCY:
        return f"{ticker}.L", 0.01
    if "." in ticker:
        return ticker.replace(".", "-"), 1.0
    return ticker, 1.0
