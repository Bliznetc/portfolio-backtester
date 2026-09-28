"""
Storage for imported broker transactions (buys, sells, dividends), scoped
per saved portfolio config so stats for different brokers/accounts never mix.
"""

import json
import logging
from typing import List

from . import db

logger = logging.getLogger(__name__)


def upsert_transactions(portfolio_config_id: int, rows: List[dict]) -> int:
    """
    Bulk-insert transactions, skipping any that already exist for this
    portfolio (same broker_tx_id) - safe to call repeatedly with the same
    export file, which is what makes re-uploading a monthly statement safe.

    Returns the number of rows actually inserted (new transactions).
    """
    if not rows:
        return 0

    values = [
        (
            portfolio_config_id,
            r["broker"],
            r["broker_tx_id"],
            r["action"],
            r["ticker"],
            r.get("isin"),
            r["executed_at"],
            r.get("quantity"),
            r.get("price"),
            r.get("price_currency"),
            r.get("result"),
            r.get("result_currency"),
            r.get("total"),
            r.get("total_currency"),
            json.dumps(r["raw"], default=str),
        )
        for r in rows
    ]

    inserted = db.execute_values(
        """
        INSERT INTO transactions (
            portfolio_config_id, broker, broker_tx_id, action, ticker, isin,
            executed_at, quantity, price, price_currency, result,
            result_currency, total, total_currency, raw
        )
        VALUES %s
        ON CONFLICT (portfolio_config_id, broker_tx_id) DO NOTHING
        RETURNING id
        """,
        values,
        fetch=True,
    )
    count = len(inserted)
    logger.info(f"Inserted {count}/{len(rows)} new transactions for portfolio {portfolio_config_id}")
    return count


def list_transactions(portfolio_config_id: int) -> List[dict]:
    """All transactions for a portfolio config, oldest first."""
    return db.execute(
        """
        SELECT * FROM transactions
        WHERE portfolio_config_id = %s
        ORDER BY executed_at
        """,
        (portfolio_config_id,),
        fetch=True,
    )


def delete_transactions(portfolio_config_id: int) -> None:
    """Remove all imported transactions for a portfolio config."""
    db.execute(
        "DELETE FROM transactions WHERE portfolio_config_id = %s",
        (portfolio_config_id,),
    )
