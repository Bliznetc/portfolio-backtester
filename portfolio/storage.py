"""
Persistent storage for saved portfolios and imported transactions.

Uses SQLAlchemy Core so the same code runs against:
- Postgres (e.g. Supabase) when DATABASE_URL is configured
- a local SQLite file otherwise (no setup, no credentials)
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table,
    UniqueConstraint, create_engine, delete, insert, select, update,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "default"
DEFAULT_SQLITE_PATH = Path(__file__).parent.parent / "data" / "portfolios.db"

metadata = MetaData()

portfolios_table = Table(
    "portfolios", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False, default=DEFAULT_USER_ID),
    Column("name", String(200), nullable=False),
    Column("broker", String(32), nullable=False, default="manual"),
    Column("base_currency", String(3), nullable=False, default="USD"),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    UniqueConstraint("user_id", "name", name="uq_portfolio_user_name"),
)

portfolio_weights_table = Table(
    "portfolio_weights", metadata,
    Column("portfolio_id", Integer, ForeignKey("portfolios.id"), primary_key=True),
    Column("ticker", String(32), primary_key=True),
    Column("weight", Float, nullable=False),
)

transactions_table = Table(
    "transactions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("portfolio_id", Integer, ForeignKey("portfolios.id"), nullable=False),
    Column("broker_tx_id", String(128), nullable=False),
    Column("date", DateTime, nullable=False),
    Column("type", String(32), nullable=False),
    Column("ticker", String(32)),
    Column("isin", String(12)),
    Column("name", String(200)),
    Column("quantity", Float),
    Column("price", Float),
    Column("currency", String(3)),
    Column("fx_rate", Float),
    Column("total", Float),
    Column("fees", Float),
    Column("notes", String(500)),
    UniqueConstraint("portfolio_id", "broker_tx_id", name="uq_tx_portfolio_broker_id"),
)


@dataclass
class PortfolioRecord:
    """A saved portfolio and its target weights."""
    id: int
    name: str
    broker: str
    base_currency: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class Transaction:
    """A single broker transaction in the normalized, broker-independent format."""
    broker_tx_id: str
    date: datetime
    type: str  # e.g. 'buy', 'sell', 'dividend', 'deposit', 'withdrawal', 'fee'
    ticker: Optional[str] = None
    isin: Optional[str] = None
    name: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    fx_rate: Optional[float] = None
    total: Optional[float] = None
    fees: Optional[float] = None
    notes: Optional[str] = None


def get_database_url() -> str:
    """
    Resolve the database URL.

    Order: Streamlit secrets -> DATABASE_URL env var -> local SQLite file.
    """
    try:
        import streamlit as st
        url = st.secrets.get("DATABASE_URL")
        if url:
            return url
    except Exception:
        # No secrets file or not running under Streamlit
        pass

    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


def _with_postgres_driver(url: str) -> str:
    """Pin Postgres URLs to the psycopg (v3) driver, whatever form the provider gave."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PortfolioStore:
    """CRUD access to saved portfolios and their transactions."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or get_database_url()
        self.engine: Engine = create_engine(_with_postgres_driver(url), pool_pre_ping=True)
        metadata.create_all(self.engine)
        logger.info(f"PortfolioStore connected ({self.engine.dialect.name})")

    # ------------------------------------------------------------------ #
    # Portfolios
    # ------------------------------------------------------------------ #

    def list_portfolios(self, user_id: str = DEFAULT_USER_ID) -> List[PortfolioRecord]:
        """Return all portfolios for a user (without weights), ordered by name."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(portfolios_table)
                .where(portfolios_table.c.user_id == user_id)
                .order_by(portfolios_table.c.name)
            ).mappings().all()
        return [PortfolioRecord(**row) for row in rows]

    def get_portfolio(self, portfolio_id: int) -> Optional[PortfolioRecord]:
        """Return a portfolio with its weights, or None if it does not exist."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(portfolios_table).where(portfolios_table.c.id == portfolio_id)
            ).mappings().first()
            if row is None:
                return None
            weight_rows = conn.execute(
                select(portfolio_weights_table.c.ticker, portfolio_weights_table.c.weight)
                .where(portfolio_weights_table.c.portfolio_id == portfolio_id)
            ).all()
        return PortfolioRecord(**row, weights={t: w for t, w in weight_rows})

    def get_portfolio_by_name(
        self, name: str, user_id: str = DEFAULT_USER_ID
    ) -> Optional[PortfolioRecord]:
        with self.engine.connect() as conn:
            portfolio_id = conn.execute(
                select(portfolios_table.c.id).where(
                    portfolios_table.c.user_id == user_id,
                    portfolios_table.c.name == name,
                )
            ).scalar()
        return self.get_portfolio(portfolio_id) if portfolio_id is not None else None

    def save_portfolio(
        self,
        name: str,
        weights: Dict[str, float],
        broker: str = "manual",
        base_currency: str = "USD",
        user_id: str = DEFAULT_USER_ID,
    ) -> int:
        """
        Create a portfolio, or replace the weights of an existing one with the same name.

        Returns:
            The portfolio id
        """
        name = name.strip()
        if not name:
            raise ValueError("Portfolio name cannot be empty")

        now = _utcnow()
        with self.engine.begin() as conn:
            portfolio_id = conn.execute(
                select(portfolios_table.c.id).where(
                    portfolios_table.c.user_id == user_id,
                    portfolios_table.c.name == name,
                )
            ).scalar()

            if portfolio_id is None:
                portfolio_id = conn.execute(
                    insert(portfolios_table).values(
                        user_id=user_id, name=name, broker=broker,
                        base_currency=base_currency, created_at=now, updated_at=now,
                    )
                ).inserted_primary_key[0]
            else:
                conn.execute(
                    update(portfolios_table)
                    .where(portfolios_table.c.id == portfolio_id)
                    .values(updated_at=now)
                )

            self._replace_weights(conn, portfolio_id, weights)
        return portfolio_id

    def update_weights(self, portfolio_id: int, weights: Dict[str, float]) -> None:
        """Replace the weights of an existing portfolio."""
        with self.engine.begin() as conn:
            conn.execute(
                update(portfolios_table)
                .where(portfolios_table.c.id == portfolio_id)
                .values(updated_at=_utcnow())
            )
            self._replace_weights(conn, portfolio_id, weights)

    def delete_portfolio(self, portfolio_id: int) -> None:
        """Delete a portfolio together with its weights and transactions."""
        with self.engine.begin() as conn:
            conn.execute(delete(portfolio_weights_table).where(
                portfolio_weights_table.c.portfolio_id == portfolio_id))
            conn.execute(delete(transactions_table).where(
                transactions_table.c.portfolio_id == portfolio_id))
            conn.execute(delete(portfolios_table).where(
                portfolios_table.c.id == portfolio_id))

    @staticmethod
    def _replace_weights(conn, portfolio_id: int, weights: Dict[str, float]) -> None:
        conn.execute(delete(portfolio_weights_table).where(
            portfolio_weights_table.c.portfolio_id == portfolio_id))
        rows = [
            {"portfolio_id": portfolio_id, "ticker": ticker, "weight": float(weight)}
            for ticker, weight in weights.items()
        ]
        if rows:
            conn.execute(insert(portfolio_weights_table), rows)

    # ------------------------------------------------------------------ #
    # Transactions
    # ------------------------------------------------------------------ #

    def add_transactions(
        self, portfolio_id: int, transactions: List[Transaction]
    ) -> Tuple[int, int]:
        """
        Insert transactions, skipping any whose broker_tx_id is already stored.

        Re-uploading the same (or an overlapping) export is therefore safe.

        Returns:
            (inserted_count, skipped_count)
        """
        with self.engine.begin() as conn:
            existing = set(conn.execute(
                select(transactions_table.c.broker_tx_id)
                .where(transactions_table.c.portfolio_id == portfolio_id)
            ).scalars())

            rows = []
            for tx in transactions:
                if tx.broker_tx_id in existing:
                    continue
                existing.add(tx.broker_tx_id)
                rows.append({"portfolio_id": portfolio_id, **tx.__dict__})

            if rows:
                conn.execute(insert(transactions_table), rows)
                conn.execute(
                    update(portfolios_table)
                    .where(portfolios_table.c.id == portfolio_id)
                    .values(updated_at=_utcnow())
                )
        return len(rows), len(transactions) - len(rows)

    def list_transactions(
        self, portfolio_id: int, ticker: Optional[str] = None
    ) -> List[Transaction]:
        """Return a portfolio's transactions in chronological order."""
        query = select(transactions_table).where(
            transactions_table.c.portfolio_id == portfolio_id)
        if ticker is not None:
            query = query.where(transactions_table.c.ticker == ticker)
        query = query.order_by(transactions_table.c.date, transactions_table.c.id)

        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [
            Transaction(**{k: v for k, v in row.items() if k not in ("id", "portfolio_id")})
            for row in rows
        ]
