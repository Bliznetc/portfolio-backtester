"""
FIFO-based realized/unrealized profit & loss, computed independently from
whatever a broker reports - lets you sanity-check Trading 212's own
`result` figures, and is the only source of realized P&L at all for
Revolut, whose statement doesn't report one.

Corporate actions (merger conversions, stock splits - see
core/positions.py) change share count without a cost basis we can assign:
the data doesn't say which "shares removed" row pairs with which "shares
added" row, or at what cost. Those shares are tracked separately as
"cost basis unknown" and excluded from P&L math rather than guessed at.
Selling more than the recorded purchase history covers (e.g. a buy that
happened before the export's start date) is treated the same way.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class _Lot:
    quantity: float
    cost_price: Optional[float]  # None = cost basis unknown
    currency: Optional[str]


@dataclass
class TickerPnL:
    """FIFO P&L for one ticker within a portfolio."""
    ticker: str
    # currency -> amount; sells can settle in different currencies (see
    # positions.py), so this stays unconverted like everywhere else here.
    realized_pnl: Dict[str, float] = field(default_factory=dict)
    unrealized_pnl: Optional[float] = None       # in price_currency; None if no current price given
    cost_basis_remaining: Optional[float] = None  # in price_currency; None if nothing with known cost is held
    # Average price of the shares still held (cost_basis_remaining /
    # quantity_known_cost) - deliberately NOT the same as positions.py's
    # avg_buy_price, which averages every buy ever made including shares
    # long since sold at a different price.
    avg_cost_price: Optional[float] = None
    quantity_known_cost: float = 0.0
    quantity_unknown_cost: float = 0.0            # from corporate actions or missing purchase history


def compute_fifo_pnl(
    transactions: List[dict],
    current_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, TickerPnL]:
    """
    Replay each ticker's buys/sells/adjustments in chronological order,
    matching sells against the oldest open lots first (FIFO), and return
    per-ticker realized P&L plus unrealized P&L for whatever's still held
    (only computed for tickers present in `current_prices`).
    """
    current_prices = current_prices or {}
    by_ticker = defaultdict(list)
    for t in transactions:
        by_ticker[t["ticker"]].append(t)

    results = {}
    for ticker, rows in by_ticker.items():
        rows_sorted = sorted(rows, key=lambda r: r["executed_at"])
        lots: deque = deque()
        realized: Dict[str, float] = defaultdict(float)
        unknown_cost_qty = 0.0

        for r in rows_sorted:
            action = r["action"]
            qty = r.get("quantity") or 0.0

            if action == "buy":
                lots.append(_Lot(qty, r.get("price"), r.get("price_currency")))
                continue

            if action == "sell":
                sell_price = r.get("price")
                remaining = qty
                while remaining > 1e-9 and lots:
                    lot = lots[0]
                    consumed = min(lot.quantity, remaining)
                    if lot.cost_price is not None and sell_price is not None:
                        pnl_ccy = r.get("price_currency") or lot.currency
                        realized[pnl_ccy] += consumed * (sell_price - lot.cost_price)
                    else:
                        unknown_cost_qty += consumed
                    lot.quantity -= consumed
                    remaining -= consumed
                    if lot.quantity <= 1e-9:
                        lots.popleft()
                if remaining > 1e-9:
                    # Sold more than recorded buys cover (e.g. purchased
                    # before the export's start date) - no cost basis for
                    # this portion, so it's excluded from realized P&L.
                    unknown_cost_qty += remaining
                continue

            if action == "adjustment":
                if qty > 0:
                    lots.append(_Lot(qty, None, r.get("price_currency")))
                else:
                    remaining = -qty
                    while remaining > 1e-9 and lots:
                        lot = lots[0]
                        consumed = min(lot.quantity, remaining)
                        lot.quantity -= consumed
                        remaining -= consumed
                        if lot.quantity <= 1e-9:
                            lots.popleft()
                    # Any leftover `remaining` here means more was removed
                    # than was on record - nothing meaningful to do with it.
                continue

        cost_basis_remaining = sum(l.quantity * l.cost_price for l in lots if l.cost_price is not None)
        known_qty = sum(l.quantity for l in lots if l.cost_price is not None)
        unknown_qty = sum(l.quantity for l in lots if l.cost_price is None) + unknown_cost_qty

        current_price = current_prices.get(ticker)
        unrealized = (known_qty * current_price - cost_basis_remaining) if current_price is not None else None
        avg_cost_price = (cost_basis_remaining / known_qty) if known_qty > 1e-9 else None

        results[ticker] = TickerPnL(
            ticker=ticker,
            realized_pnl=dict(realized),
            unrealized_pnl=unrealized,
            cost_basis_remaining=cost_basis_remaining if known_qty > 1e-9 else None,
            avg_cost_price=avg_cost_price,
            quantity_known_cost=known_qty,
            quantity_unknown_cost=unknown_qty,
        )

    return results
