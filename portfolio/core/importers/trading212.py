"""
Trading 212 CSV export importer.

Normalizes a Trading 212 "transaction history" export into the common
transaction shape used by transactions_store: one dict per buy/sell/dividend
row, with a stable broker_tx_id so re-uploading the same (or an overlapping)
monthly export doesn't create duplicates.

Trading 212 exports mix real trades with account/card activity (Deposit,
Withdrawal, Card debit/credit, Interest on cash, Spending cashback, Currency
conversion, New card cost) - only stock trades and dividends are imported;
everything else is skipped since it doesn't affect stock positions.
"""

import csv
import hashlib
import io
import logging
from datetime import datetime
from typing import List, Union

logger = logging.getLogger(__name__)

BROKER = "trading212"


def _classify_action(action_raw: str) -> str:
    lowered = action_raw.lower()
    if "buy" in lowered:
        return "buy"
    if "sell" in lowered:
        return "sell"
    if lowered.startswith("dividend"):
        return "dividend"
    return ""  # cash/card/other row - not a position, caller skips it


def _parse_float(value):
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def _parse_time(value: str) -> datetime:
    # Trading 212 timestamps look like "2025-09-16 10:50:21+00:00"
    return datetime.fromisoformat(value.strip())


def _normalize_price(price, currency):
    """Trading 212 quotes UK-listed shares in GBX (pence); convert to GBP."""
    if currency == "GBX" and price is not None:
        return price / 100.0, "GBP"
    return price, currency


def _broker_tx_id(row: dict, action_raw: str, ticker: str, isin, time_str: str,
                   quantity, price, total) -> str:
    raw_id = (row.get("ID") or "").strip()
    if raw_id:
        return raw_id
    # Dividend rows (and possibly others) have no ID from Trading 212 - derive
    # a stable one from fields that together identify this exact row, so the
    # same dividend hashes identically every time this file is re-uploaded.
    key = "|".join(str(x) for x in (action_raw, time_str, isin, ticker, quantity, price, total))
    return "h_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def parse(file: Union[str, bytes, "io.IOBase"]) -> List[dict]:
    """
    Parse a Trading 212 "transaction history" CSV export.

    Returns a list of dicts matching the shape transactions_store.upsert_transactions
    expects: broker, broker_tx_id, action, ticker, isin, executed_at, quantity,
    price, price_currency, result, result_currency, total, total_currency, raw.
    """
    if hasattr(file, "read"):
        content = file.read()
    else:
        content = file
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(content))

    transactions = []
    skipped = 0

    for row in reader:
        action_raw = (row.get("Action") or "").strip()
        action = _classify_action(action_raw)
        if not action:
            skipped += 1
            continue

        ticker = (row.get("Ticker") or "").strip()
        if not ticker:
            skipped += 1
            continue

        isin = (row.get("ISIN") or "").strip() or None
        time_str = (row.get("Time (UTC)") or "").strip()
        executed_at = _parse_time(time_str)

        quantity = _parse_float(row.get("No. of shares"))
        price = _parse_float(row.get("Price / share"))
        price_currency = (row.get("Currency (Price / share)") or "").strip() or None
        price, price_currency = _normalize_price(price, price_currency)

        result = _parse_float(row.get("Result"))
        result_currency = (row.get("Currency (Result)") or "").strip() or None

        total = _parse_float(row.get("Total"))
        total_currency = (row.get("Currency (Total)") or "").strip() or None

        broker_tx_id = _broker_tx_id(row, action_raw, ticker, isin, time_str, quantity, price, total)

        transactions.append({
            "broker": BROKER,
            "broker_tx_id": broker_tx_id,
            "action": action,
            "ticker": ticker,
            "isin": isin,
            "executed_at": executed_at,
            "quantity": quantity,
            "price": price,
            "price_currency": price_currency,
            "result": result,
            "result_currency": result_currency,
            "total": total,
            "total_currency": total_currency,
            "raw": row,
        })

    logger.info(f"Parsed {len(transactions)} trade/dividend rows, skipped {skipped} non-position rows")
    return transactions
