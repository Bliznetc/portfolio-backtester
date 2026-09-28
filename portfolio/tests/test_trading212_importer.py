#!/usr/bin/env python3
"""
Standalone check for the Trading 212 importer and positions calculator.
No database required - pure parsing/computation, run with:
    python3 portfolio/tests/test_trading212_importer.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.core.importers import trading212
from portfolio.core import positions

# A representative slice of a real Trading 212 "transaction history" export:
# cash/card activity that should be skipped, a GBX-quoted (pence) UK stock
# that needs /100 conversion, a dividend row (no source ID - broker_tx_id
# must be derived), and a buy/sell pair to exercise positions math.
SAMPLE_CSV = """Action,Time (UTC),ISIN,Ticker,Name,Notes,ID,No. of shares,Price / share,Currency (Price / share),Exchange rate,Result,Currency (Result),Total,Currency (Total),Withholding tax,Currency (Withholding tax),Stamp duty reserve tax,Currency (Stamp duty reserve tax),Currency conversion from amount,Currency (Currency conversion from amount),Currency conversion to amount,Currency (Currency conversion to amount),Currency conversion fee,Currency (Currency conversion fee),Merchant name,Merchant category
Deposit,2025-09-13 21:15:46+00:00,,,,"Transaction ID: HDKKGV8C6X4ZH2G3",f67f66ff-e1f7-44a5-b2f8-7e83f6e51fdb,,,,,,,320.00,"PLN",,,,,,,,,,,,
Market buy,2025-09-16 10:50:21+00:00,US67066G1040,NVDA,"Nvidia",,EOF38850966465,1.0000000000,177.1700000000,USD,1.00000000,,,177.17,"USD",,,,,,,,,,,,
Market buy,2026-05-26 10:03:47+00:00,GB0007980591,BP,"BP",,EOF51659754256,2.7326007300,546.0000000000,GBX,100.00000000,,,15.00,"GBP",,,0.08,"GBP",,,,,,,,
Dividend (Dividend),2025-09-30 13:12:41+00:00,US11135F1012,AVGO,"Broadcom",,,1.0000000000,0.501500,USD,3.63460000,,,1.82,"PLN",0.09,USD,,,,,,,,,,
Market sell,2026-01-02 22:46:13+00:00,US5951121038,MU,"Micron Technology",,EOF44316297592,1.0000000000,317.5200000000,USD,1.00000000,157.82,"USD",317.52,"USD",,,,,,,,,,,,
Market buy,2025-09-16 13:30:08+00:00,US5951121038,MU,"Micron Technology",,EOF38853571437,1.0000000000,300.0000000000,USD,1.00000000,,,300.00,"USD",,,,,,,,,,,,
Interest on cash,2025-09-17 00:16:46+00:00,,,,"Interest on cash",72b1ec4e-90f1-4996-b87e-2bfdbdceb8e5,,,,,,,0.35,"PLN",,,,,,,,,,,,
Card debit,2025-09-22 21:24:27+00:00,,,,,fce42f40-5b7d-4078-aadf-b22e7d846197,,,,,,,-66.13,"USD",,,,,,,,,,,"From You Flowers","MISCELLANEOUS"
"""


def main():
    transactions = trading212.parse(SAMPLE_CSV)

    # 5 trade/dividend rows; 3 cash/card rows skipped
    assert len(transactions) == 5, f"expected 5 rows, got {len(transactions)}"
    print(f"✓ parsed {len(transactions)} trade/dividend rows, skipped 3 cash/card rows")

    by_ticker = {(t["ticker"], t["action"]): t for t in transactions}

    bp_buy = by_ticker[("BP", "buy")]
    assert bp_buy["price_currency"] == "GBP", "GBX should be normalized to GBP"
    assert abs(bp_buy["price"] - 5.46) < 1e-9, "546 GBX should become 5.46 GBP"
    print("✓ GBX (pence) correctly normalized to GBP")

    dividend = by_ticker[("AVGO", "dividend")]
    assert dividend["broker_tx_id"].startswith("h_"), "dividend rows have no source ID, need a derived one"
    print("✓ dividend row (no source ID) got a derived broker_tx_id")

    # Re-parsing the same content must produce identical IDs - this is what
    # makes re-uploading an overlapping monthly export a safe no-op.
    transactions_again = trading212.parse(SAMPLE_CSV)
    ids_first = [t["broker_tx_id"] for t in transactions]
    ids_second = [t["broker_tx_id"] for t in transactions_again]
    assert ids_first == ids_second, "broker_tx_id must be stable across re-parses of the same file"
    assert len(set(ids_first)) == len(ids_first), "broker_tx_ids must be unique within one parse"
    print("✓ broker_tx_id stable across re-parses and unique within a parse")

    computed = positions.compute_positions(transactions)

    mu = computed["MU"]
    assert mu.quantity_held == 0.0, "bought 1, sold 1 -> 0 held"
    assert mu.realized_profit == {"USD": 157.82}
    print("✓ MU: fully sold, realized profit matches broker-reported Result")

    avgo = computed["AVGO"]
    assert avgo.dividends_received == {"PLN": 1.82}
    print("✓ AVGO: dividend recorded in its settlement currency (PLN), unconverted")

    current_prices = {"NVDA": 200.0, "MU": 350.0}  # BP deliberately left unpriced
    weights, total_value, unpriced = positions.compute_weights(computed, current_prices)
    assert "BP" in unpriced, "BP has no current price and no holding to weight anyway"
    assert set(weights) == {"NVDA"}, f"only NVDA has quantity_held > 0 and a price, got {weights}"
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    print("✓ compute_weights: unpriced/zero-holding tickers excluded, remaining weights sum to 1")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
