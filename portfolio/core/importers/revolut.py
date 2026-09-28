"""
Revolut "Account Statement" PDF importer.

Revolut exports a PDF report (not CSV), with a section per currency the
account holds a balance in - e.g. "USD Portfolio breakdown" / "USD
Transactions", then the same again for "EUR", etc. This is parsed generically
from whichever ISO currency code each section header names, so it isn't
hardcoded to USD/EUR - a GBP or CHF section works the same way.

pdfplumber's table-detection heuristics turned out to misalign columns on
this document (column positions shift depending on what else shares the
page), so this parses plain extracted text with regexes instead - each
transaction row has a rigid, predictable format.

Only trade, dividend, and corporate-action (merger/stock-split) rows are
imported; cash top-ups/withdrawals and card rewards are skipped since they
don't affect stock positions.
"""

import hashlib
import io
import logging
import re
from datetime import datetime
from typing import List, Union

logger = logging.getLogger(__name__)

BROKER = "revolut"

# Currency symbol is whatever non-digit, non-space token precedes a number
# (US$, €, £, ...). We don't need to interpret it - the section header
# already gives us the ISO currency code generically.
_CCY_SYM = r'[^\d\s]+'

_SECTION_RE = re.compile(r'^([A-Z]{3})\s+(Account summary|Portfolio breakdown|Transactions)$')
_TIMESTAMP_RE = re.compile(r'^(\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} GMT)\s+(.*)$')
_ISIN_RE = re.compile(r'\b([A-Z]{2}[A-Z0-9]{9}\d)\b')

_TRADE_RE = re.compile(
    r'^(?P<symbol>[A-Z][A-Z0-9.]*)\s+Trade\s*-\s*(?:Limit|Market)\s+'
    rf'(?P<quantity>[\d,]+\.?\d*)\s+{_CCY_SYM}(?P<price>[\d,]+\.?\d*)\s+'
    rf'(?P<side>Buy|Sell)\s+{_CCY_SYM}(?P<value>[\d,]+\.?\d*)\s+'
    rf'{_CCY_SYM}(?P<fees>[\d,]+\.?\d*)\s+{_CCY_SYM}(?P<commission>[\d,]+\.?\d*)$'
)
_DIVIDEND_RE = re.compile(
    r'^(?P<symbol>[A-Z][A-Z0-9.]*)\s+Dividend\s+'
    rf'{_CCY_SYM}(?P<value>[\d,]+\.?\d*)\s+'
    rf'{_CCY_SYM}(?P<fees>[\d,]+\.?\d*)\s+{_CCY_SYM}(?P<commission>[\d,]+\.?\d*)$'
)
# Corporate actions (merger conversions, stock splits): signed quantity,
# no price - e.g. "CIVI Merger - stock -0.00406015 US$0 US$0 US$0" when a
# holding gets converted away, or a positive quantity for the shares it
# converts into.
_ADJUSTMENT_RE = re.compile(
    r'^(?P<symbol>[A-Z][A-Z0-9.]*)\s+(?:Merger - stock|Stock split)\s+'
    rf'(?P<quantity>-?[\d,]+\.?\d*)\s+{_CCY_SYM}[\d,]+\.?\d*\s+'
    rf'{_CCY_SYM}[\d,]+\.?\d*\s+{_CCY_SYM}[\d,]+\.?\d*$'
)


def _parse_amount(value: str) -> float:
    return float(value.replace(",", ""))


def _broker_tx_id(action: str, ccy: str, timestamp: str, symbol: str, quantity, price, value) -> str:
    # Revolut's statement has no per-row transaction ID at all (unlike
    # Trading 212), so every row needs a derived key. Timestamps are unique
    # to the second, so (action, currency, timestamp, symbol, quantity,
    # price, value) together reliably identify one row, keeping re-uploads
    # of the same or an overlapping statement idempotent.
    key = "|".join(str(x) for x in (action, ccy, timestamp, symbol, quantity, price, value))
    return "h_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def parse(file: Union[str, bytes, "io.IOBase"]) -> List[dict]:
    """
    Parse a Revolut "Account Statement" PDF export.

    Returns a list of dicts matching the shape transactions_store.upsert_transactions
    expects: broker, broker_tx_id, action, ticker, isin, executed_at, quantity,
    price, price_currency, result, result_currency, total, total_currency, raw.
    action is one of 'buy', 'sell', 'dividend', or 'adjustment' (a merger
    conversion or stock split - quantity is signed, price is always None).
    """
    import pdfplumber

    with pdfplumber.open(file) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    return parse_text(full_text)


def parse_text(full_text: str) -> List[dict]:
    """
    Parse already-extracted plain text from a Revolut statement (the actual
    line-parsing logic, split out from the PDF-extraction step above so it
    can be unit-tested with a plain string fixture, no real PDF required).
    """
    section = None  # (kind, ccy) - kind is 'Account summary' | 'Portfolio breakdown' | 'Transactions'
    isin_by_symbol = {}
    transactions = []
    skipped = 0

    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "Glossary":
            section = None  # end of a statement's content; footer boilerplate follows
            continue

        section_m = _SECTION_RE.match(line)
        if section_m:
            section = (section_m.group(2), section_m.group(1))
            continue
        if section is None:
            continue
        kind, ccy = section

        if kind == "Portfolio breakdown":
            isin_m = _ISIN_RE.search(line)
            if isin_m:
                isin_by_symbol[line.split()[0]] = isin_m.group(1)
            continue

        if kind != "Transactions":
            continue

        ts_m = _TIMESTAMP_RE.match(line)
        if not ts_m:
            continue  # header row ("Date Symbol Type ...") or similar, not a data row
        timestamp_str, rest = ts_m.groups()
        executed_at = datetime.strptime(timestamp_str, "%d %b %Y %H:%M:%S GMT")

        trade_m = _TRADE_RE.match(rest)
        if trade_m:
            symbol = trade_m["symbol"]
            quantity = _parse_amount(trade_m["quantity"])
            price = _parse_amount(trade_m["price"])
            value = _parse_amount(trade_m["value"])
            transactions.append({
                "broker": BROKER,
                "broker_tx_id": _broker_tx_id("trade", ccy, timestamp_str, symbol, quantity, price, value),
                "action": "buy" if trade_m["side"] == "Buy" else "sell",
                "ticker": symbol,
                "isin": isin_by_symbol.get(symbol),
                "executed_at": executed_at,
                "quantity": quantity,
                "price": price,
                "price_currency": ccy,
                "result": None,
                "result_currency": None,
                "total": value,
                "total_currency": ccy,
                "raw": {"line": line, "currency": ccy},
            })
            continue

        div_m = _DIVIDEND_RE.match(rest)
        if div_m:
            symbol = div_m["symbol"]
            value = _parse_amount(div_m["value"])
            transactions.append({
                "broker": BROKER,
                "broker_tx_id": _broker_tx_id("dividend", ccy, timestamp_str, symbol, None, None, value),
                "action": "dividend",
                "ticker": symbol,
                "isin": isin_by_symbol.get(symbol),
                "executed_at": executed_at,
                "quantity": None,
                "price": None,
                "price_currency": None,
                "result": None,
                "result_currency": None,
                "total": value,
                "total_currency": ccy,
                "raw": {"line": line, "currency": ccy},
            })
            continue

        adj_m = _ADJUSTMENT_RE.match(rest)
        if adj_m:
            symbol = adj_m["symbol"]
            quantity = _parse_amount(adj_m["quantity"])
            transactions.append({
                "broker": BROKER,
                "broker_tx_id": _broker_tx_id("adjustment", ccy, timestamp_str, symbol, quantity, None, None),
                "action": "adjustment",
                "ticker": symbol,
                "isin": isin_by_symbol.get(symbol),
                "executed_at": executed_at,
                "quantity": quantity,
                "price": None,
                "price_currency": ccy,
                "result": None,
                "result_currency": None,
                "total": None,
                "total_currency": None,
                "raw": {"line": line, "currency": ccy},
            })
            continue

        skipped += 1  # cash top-up/withdrawal, reward, or other non-position activity

    logger.info(f"Parsed {len(transactions)} trade/dividend/adjustment rows, skipped {skipped} non-position rows")
    return transactions
