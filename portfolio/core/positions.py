"""
Computes current holdings and per-ticker stats from a portfolio's imported
transaction history. Pure computation - no database or network access.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TickerPosition:
    """Current holding and lifetime stats for one ticker within a portfolio."""
    ticker: str
    quantity_held: float
    isin: Optional[str]
    price_currency: Optional[str]  # currency of avg_buy_price / avg_sell_price
    avg_buy_price: Optional[float]
    avg_sell_price: Optional[float]
    total_bought_qty: float
    total_sold_qty: float
    num_transactions: int
    # currency -> amount; kept unconverted since a ticker's sells/dividends
    # can legitimately settle in different currencies (e.g. AutoInvest
    # rebalances settle in account currency, manual trades in the trade's
    # own currency) and picking an FX rate to unify them would be a guess.
    realized_profit: Dict[str, float] = field(default_factory=dict)
    dividends_received: Dict[str, float] = field(default_factory=dict)


def compute_positions(transactions: List[dict]) -> Dict[str, TickerPosition]:
    """
    Group a portfolio's transactions by ticker and compute current holdings
    and stats for each. `transactions` uses the row shape produced by an
    importer (or read back from transactions_store.list_transactions).
    """
    by_ticker = defaultdict(list)
    for t in transactions:
        by_ticker[t["ticker"]].append(t)

    positions = {}
    for ticker, rows in by_ticker.items():
        buys = [r for r in rows if r["action"] == "buy"]
        sells = [r for r in rows if r["action"] == "sell"]
        dividends = [r for r in rows if r["action"] == "dividend"]
        # Corporate actions (merger conversions, stock splits): a signed
        # quantity change with no price - e.g. a merger removes shares from
        # the old symbol (negative) and adds shares to the new one
        # (positive). Folded into quantity_held directly, kept out of
        # avg_buy_price/avg_sell_price since there's no real trade price.
        adjustments = [r for r in rows if r["action"] == "adjustment"]

        total_bought_qty = sum(r["quantity"] or 0 for r in buys)
        total_sold_qty = sum(r["quantity"] or 0 for r in sells)
        net_adjustment_qty = sum(r["quantity"] or 0 for r in adjustments)

        price_currency = next((r["price_currency"] for r in rows if r.get("price_currency")), None)
        isin = next((r["isin"] for r in rows if r.get("isin")), None)

        positions[ticker] = TickerPosition(
            ticker=ticker,
            quantity_held=total_bought_qty - total_sold_qty + net_adjustment_qty,
            isin=isin,
            price_currency=price_currency,
            avg_buy_price=_weighted_avg_price(buys),
            avg_sell_price=_weighted_avg_price(sells),
            total_bought_qty=total_bought_qty,
            total_sold_qty=total_sold_qty,
            num_transactions=len(rows),
            realized_profit=_sum_by_currency(sells, "result", "result_currency"),
            dividends_received=_sum_by_currency(dividends, "total", "total_currency"),
        )

    return positions


def compute_weights(
    positions: Dict[str, TickerPosition],
    current_prices: Dict[str, float],
) -> Tuple[Dict[str, float], float, List[str]]:
    """
    Convert holdings into normalized weights using current prices. Prices
    must already be in one common currency (the caller's responsibility -
    mixing currencies here would silently misstate proportions rather than
    raise, so we don't attempt any FX conversion ourselves).

    Tickers with no held quantity, or missing from current_prices, are
    excluded from the weights and returned in `unpriced` so the caller can
    surface that (e.g. a UK-listed ticker current price wasn't fetchable).

    Returns (weights, total_value, unpriced_tickers).
    """
    values = {}
    unpriced = []
    for ticker, pos in positions.items():
        if pos.quantity_held <= 0:
            continue
        price = current_prices.get(ticker)
        if price is None:
            unpriced.append(ticker)
            continue
        values[ticker] = pos.quantity_held * price

    total_value = sum(values.values())
    if total_value <= 0:
        return {}, 0.0, unpriced

    weights = {ticker: value / total_value for ticker, value in values.items()}
    return weights, total_value, unpriced


def _weighted_avg_price(rows: List[dict]) -> Optional[float]:
    total_qty = 0.0
    total_cost = 0.0
    for r in rows:
        qty, price = r.get("quantity"), r.get("price")
        if qty is None or price is None:
            continue
        total_qty += qty
        total_cost += qty * price
    if total_qty == 0:
        return None
    return total_cost / total_qty


def _sum_by_currency(rows: List[dict], amount_field: str, currency_field: str) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for r in rows:
        amount = r.get(amount_field)
        currency = r.get(currency_field)
        if amount is None or not currency:
            continue
        totals[currency] += amount
    return dict(totals)
