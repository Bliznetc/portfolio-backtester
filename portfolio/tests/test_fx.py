#!/usr/bin/env python3
"""
Standalone check for FX rate fetching and conversion. Hits the network
(yfinance) - run with: python3 portfolio/tests/test_fx.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.core import fx


def main():
    assert fx.get_rate_to_usd("USD") == 1.0
    print("✓ USD -> USD is exactly 1.0, no network call needed")

    for ccy in ["EUR", "GBP", "PLN"]:
        rate = fx.get_rate_to_usd(ccy)
        assert rate is not None and rate > 0, f"{ccy}: {rate}"
        print(f"✓ {ccy} -> USD rate fetched: {rate:.4f}")

    total = fx.convert_amounts_to_usd({"USD": 10.0, "EUR": 10.0})
    assert total is not None and total > 10.0, total  # EUR is worth more than 1 USD right now
    print(f"✓ mixed-currency dict converts to a single USD total: {total:.2f}")

    assert fx.convert_amounts_to_usd({}) == 0.0
    print("✓ empty dict converts to 0.0")

    unknown = fx.convert_amounts_to_usd({"USD": 5.0, "ZZZ": 5.0})
    assert unknown is None, unknown
    print("✓ an unresolvable currency returns None (not a silently partial sum)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
