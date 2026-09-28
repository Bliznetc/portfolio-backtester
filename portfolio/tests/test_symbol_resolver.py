#!/usr/bin/env python3
"""
Standalone check for the LSE ticker-mapping heuristic.
Run with: python3 portfolio/tests/test_symbol_resolver.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.core.symbol_resolver import resolve_yfinance_symbol


def main():
    # GB ISIN + GBP price_currency (i.e. was originally GBX pence) -> .L,
    # with a 0.01 scale since yfinance quotes LSE lines in pence, not pounds.
    symbol, scale = resolve_yfinance_symbol("BP", "GB0007980591", "GBP")
    assert symbol == "BP.L" and scale == 0.01, (symbol, scale)
    print("✓ GB ISIN + GBP (ex-GBX) ticker maps to '<ticker>.L' with a 0.01 price scale")

    # A US ticker should be left completely alone.
    symbol, scale = resolve_yfinance_symbol("AAPL", "US0378331005", "USD")
    assert symbol == "AAPL" and scale == 1.0, (symbol, scale)
    print("✓ US-domiciled ticker is left unchanged")

    # GB-domiciled but priced in USD (e.g. a Nasdaq-listed ADR) must NOT get
    # the .L suffix - GB ISIN alone isn't a reliable enough signal.
    symbol, scale = resolve_yfinance_symbol("TRMD", "GB00BZ3CNK81", "USD")
    assert symbol == "TRMD" and scale == 1.0, (symbol, scale)
    print("✓ GB ISIN priced in USD (not GBX) is left unchanged, avoiding a wrong guess")

    # No ISIN at all -> can't resolve, leave as-is.
    symbol, scale = resolve_yfinance_symbol("MYSTERY", None, "GBP")
    assert symbol == "MYSTERY" and scale == 1.0, (symbol, scale)
    print("✓ missing ISIN falls back to the plain ticker")

    # US multi-class ticker: yfinance uses a hyphen, not a period.
    symbol, scale = resolve_yfinance_symbol("BRK.B", "US0846707026", "USD")
    assert symbol == "BRK-B" and scale == 1.0, (symbol, scale)
    print("✓ 'BRK.B' maps to 'BRK-B' (yfinance's hyphen convention for share classes)")

    # Manual per-ISIN overrides (verified by hand - see module docstring)
    # take priority over the general heuristics, and never touch price_scale.
    symbol, scale = resolve_yfinance_symbol("NQSE", "IE00BYVQ9F29", "EUR")
    assert symbol == "NQSE.DE" and scale == 1.0, (symbol, scale)
    print("✓ ISIN override resolves an ETF Yahoo only lists under a different suffix")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
