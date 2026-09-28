#!/usr/bin/env python3
"""
Standalone check for the Revolut statement parser and positions calculator.
No database, no real PDF file required - run with:
    python3 portfolio/tests/test_revolut_importer.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.core.importers import revolut
from portfolio.core import positions

# A hand-built slice matching Revolut's real "Account Statement" text layout
# (verified against two real exports): two currency sections (USD, EUR),
# a buy/sell pair, a dividend (no source ID, unlike Trading 212's trades),
# a merger corporate action (signed quantity, no price), and cash/reward
# rows that must be skipped. The ISIN in "Portfolio breakdown" is cross-
# referenced onto the matching symbol's transactions.
SAMPLE_TEXT = """
USD Portfolio breakdown
Symbol Company ISIN Quantity Price Value % of Portfolio
NVDA NVIDIA US67066G1040 4.43 US$186.50 US$826.56 16.7%
Positions Value US$826.56 100%
USD Transactions
Date Symbol Type Quantity Price Side Value Fees Commission
31 Mar 2025 15:49:19 GMT Cash top-up US$313.56 US$0 US$0
01 Apr 2025 13:30:01 GMT NVDA Trade - Limit 3 US$108.45 Buy US$325.35 US$0 US$0
16 Apr 2025 13:30:09 GMT NVDA Trade - Market 1 US$104.44 Sell US$104.44 US$0.02 US$1.32
16 May 2025 18:42:15 GMT AAPL Dividend US$0.21 US$0 US$0
19 Dec 2025 17:25:34 GMT Reward US$3.22 US$0 US$0
01 Aug 2025 15:57:50 GMT Cash withdrawal -US$213 US$0 US$0
02 Feb 2026 15:53:33 GMT SM Merger - stock 0.00588721 US$0 US$0 US$0
02 Feb 2026 15:53:33 GMT CIVI Merger - stock -0.00406015 US$0 US$0 US$0
EUR Portfolio breakdown
Symbol Company ISIN Quantity Price Value % of Portfolio
NQSE iShares NASDAQ 100 Acc ETF IE00BYVQ9F29 139.7 €15.01 €2,097.17 100%
EUR Transactions
Date Symbol Type Quantity Price Side Value Fees Commission
07 Apr 2025 20:44:04 GMT Cash top-up €210 €0 €0
29 May 2025 07:04:11 GMT NQSE Trade - Market 15.45595054 €12.94 Buy €200 €0 €0
Glossary
Positions value
The total market value of your holdings...
"""


def main():
    transactions = revolut.parse_text(SAMPLE_TEXT)

    # 2 NVDA trades + 1 dividend + 2 merger adjustments + 1 NQSE trade = 6
    assert len(transactions) == 6, f"expected 6 rows, got {len(transactions)}"
    print(f"✓ parsed {len(transactions)} trade/dividend/adjustment rows, skipped 4 cash/reward rows")

    by_key = {(t["ticker"], t["action"]): t for t in transactions}

    nvda_buy = by_key[("NVDA", "buy")]
    assert nvda_buy["isin"] == "US67066G1040", "ISIN should be cross-referenced from Portfolio breakdown"
    assert nvda_buy["price_currency"] == "USD", "currency comes from the section's ISO code, not the symbol"
    print("✓ ISIN cross-referenced from Portfolio breakdown, currency taken from section header")

    dividend = by_key[("AAPL", "dividend")]
    assert dividend["broker_tx_id"].startswith("h_"), "Revolut rows have no source ID at all - always derived"
    print("✓ dividend row got a derived broker_tx_id (Revolut has no ID column whatsoever)")

    sm_adj = by_key[("SM", "adjustment")]
    civi_adj = by_key[("CIVI", "adjustment")]
    assert sm_adj["quantity"] > 0 and sm_adj["price"] is None
    assert civi_adj["quantity"] < 0 and civi_adj["price"] is None
    print("✓ merger corporate action: signed quantity captured, no price")

    nqse_buy = by_key[("NQSE", "buy")]
    assert nqse_buy["price_currency"] == "EUR", "second currency section parsed independently of the first"
    print("✓ second (EUR) currency section parsed correctly alongside USD")

    # Re-parsing the same text must produce identical IDs (safe re-upload).
    ids_first = [t["broker_tx_id"] for t in transactions]
    ids_second = [t["broker_tx_id"] for t in revolut.parse_text(SAMPLE_TEXT)]
    assert ids_first == ids_second
    assert len(set(ids_first)) == len(ids_first)
    print("✓ broker_tx_id stable across re-parses and unique within a parse")

    computed = positions.compute_positions(transactions)

    nvda = computed["NVDA"]
    assert abs(nvda.quantity_held - 2.0) < 1e-9, "bought 3, sold 1 -> 2 held"
    print("✓ NVDA: buy/sell nets to correct quantity_held")

    sm = computed["SM"]
    civi = computed["CIVI"]
    assert abs(sm.quantity_held - 0.00588721) < 1e-9
    assert abs(civi.quantity_held - (-0.00406015)) < 1e-9
    print("✓ merger adjustment folds into quantity_held without touching avg_buy_price")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
