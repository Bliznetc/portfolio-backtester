#!/usr/bin/env python3
"""
Standalone check for the FIFO P&L calculator.
Run with: python3 portfolio/tests/test_pnl.py
"""

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.core import pnl


def tx(action, ticker, dt, quantity=None, price=None, ccy="USD"):
    return {
        "action": action, "ticker": ticker,
        "executed_at": datetime.fromisoformat(dt),
        "quantity": quantity, "price": price, "price_currency": ccy,
    }


def main():
    # --- FIFO must consume the OLDEST lot first, not an average ---
    # Buy 10 @ $10, buy 10 @ $20, sell 10 -> should match the $10 lot,
    # not an average of ($10+$20)/2 = $15.
    txs = [
        tx("buy", "X", "2025-01-01", 10, 10.0),
        tx("buy", "X", "2025-02-01", 10, 20.0),
        tx("sell", "X", "2025-03-01", 10, 25.0),
    ]
    result = pnl.compute_fifo_pnl(txs)["X"]
    assert result.realized_pnl == {"USD": 150.0}, result.realized_pnl  # 10 * (25-10), not 10*(25-15)
    assert result.quantity_known_cost == 10.0  # the $20 lot is still fully held
    assert result.cost_basis_remaining == 200.0
    assert result.avg_cost_price == 20.0  # held shares only - not (10+20)/2
    print("✓ FIFO consumes the oldest lot first (not a blended average)")
    print("✓ avg_cost_price reflects only the remaining (held) lot, not lifetime buys")

    # --- Partial lot consumption splits correctly across two sells ---
    txs = [
        tx("buy", "Y", "2025-01-01", 10, 10.0),
        tx("sell", "Y", "2025-02-01", 6, 15.0),   # 6 @ $10 cost -> +30
        tx("sell", "Y", "2025-03-01", 4, 12.0),   # remaining 4 @ $10 cost -> +8
    ]
    result = pnl.compute_fifo_pnl(txs)["Y"]
    assert abs(result.realized_pnl["USD"] - 38.0) < 1e-9, result.realized_pnl
    assert result.quantity_known_cost == 0.0
    assert result.cost_basis_remaining is None
    print("✓ a lot split across multiple partial sells accumulates correctly")

    # --- Selling more than recorded buys cover: excluded, not guessed at ---
    txs = [
        tx("buy", "Z", "2025-01-01", 5, 10.0),
        tx("sell", "Z", "2025-02-01", 8, 20.0),  # only 5 have a known cost basis
    ]
    result = pnl.compute_fifo_pnl(txs)["Z"]
    assert result.realized_pnl == {"USD": 50.0}, result.realized_pnl  # only the 5 matched shares
    assert result.quantity_unknown_cost == 3.0
    print("✓ overselling beyond recorded history is excluded from P&L, not misattributed")

    # --- Corporate action: negative adjustment removes shares without realizing P&L ---
    txs = [
        tx("buy", "OLD", "2025-01-01", 10, 50.0),
        {"action": "adjustment", "ticker": "OLD", "executed_at": datetime(2025, 6, 1),
         "quantity": -10, "price": None, "price_currency": "USD"},
    ]
    result = pnl.compute_fifo_pnl(txs)["OLD"]
    assert result.realized_pnl == {}, result.realized_pnl  # a conversion, not a sale
    assert result.quantity_known_cost == 0.0
    print("✓ merger-away shares are removed via FIFO without fabricating a realized gain/loss")

    # --- Corporate action: positive adjustment creates a cost-unknown lot ---
    txs = [
        {"action": "adjustment", "ticker": "NEW", "executed_at": datetime(2025, 6, 1),
         "quantity": 3, "price": None, "price_currency": "USD"},
        tx("sell", "NEW", "2025-07-01", 3, 40.0),
    ]
    result = pnl.compute_fifo_pnl(txs)["NEW"]
    assert result.realized_pnl == {}, result.realized_pnl  # no known cost basis to compute a gain from
    assert result.quantity_unknown_cost == 3.0
    print("✓ shares received via a merger have no fabricated cost basis")

    # --- Unrealized P&L uses remaining FIFO cost basis, only when a price is given ---
    txs = [
        tx("buy", "W", "2025-01-01", 10, 10.0),
        tx("buy", "W", "2025-02-01", 10, 20.0),
        tx("sell", "W", "2025-03-01", 10, 25.0),  # consumes the $10 lot
    ]
    result_no_price = pnl.compute_fifo_pnl(txs)["W"]
    assert result_no_price.unrealized_pnl is None
    result_priced = pnl.compute_fifo_pnl(txs, current_prices={"W": 30.0})["W"]
    # 10 remaining shares at $20 cost, now worth $30 -> +100
    assert abs(result_priced.unrealized_pnl - 100.0) < 1e-9, result_priced.unrealized_pnl
    print("✓ unrealized P&L computed from the remaining FIFO lot, only when a price is supplied")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
