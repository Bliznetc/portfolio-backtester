"""
Database connection and schema management for Postgres (Neon, Supabase, or
any other Postgres host - just a connection string).

Connection string resolution order: Streamlit secrets, then DATABASE_URL
env var (so standalone scripts run via plain `python` also work).
"""

import os
import logging

import psycopg2
import psycopg2.extensions
import psycopg2.extras

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)

_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id            BIGSERIAL PRIMARY KEY,
        username      TEXT UNIQUE NOT NULL,
        name          TEXT NOT NULL,
        email         TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_configs (
        id          BIGSERIAL PRIMARY KEY,
        user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        name        TEXT NOT NULL,
        config      JSONB NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_portfolio_configs_user_id
        ON portfolio_configs(user_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS transactions (
        id                  BIGSERIAL PRIMARY KEY,
        portfolio_config_id BIGINT NOT NULL REFERENCES portfolio_configs(id) ON DELETE CASCADE,
        broker              TEXT NOT NULL,
        broker_tx_id        TEXT NOT NULL,
        action              TEXT NOT NULL,
        ticker              TEXT NOT NULL,
        isin                TEXT,
        executed_at         TIMESTAMPTZ NOT NULL,
        quantity            NUMERIC,
        price               NUMERIC,
        price_currency      TEXT,
        result              NUMERIC,
        result_currency     TEXT,
        total               NUMERIC,
        total_currency      TEXT,
        raw                 JSONB NOT NULL,
        UNIQUE (portfolio_config_id, broker_tx_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_config_id
        ON transactions(portfolio_config_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_transactions_ticker
        ON transactions(portfolio_config_id, ticker)
    """,
    """
    CREATE TABLE IF NOT EXISTS instrument_classifications (
        yf_symbol         TEXT PRIMARY KEY,
        display_ticker    TEXT,
        yf_sector         TEXT,
        yf_industry       TEXT,
        business_summary  TEXT,
        ai_classification TEXT,
        ai_rationale      TEXT,
        ai_model          TEXT,
        ai_generated_at   TIMESTAMPTZ,
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dependency_categories (
        id                      BIGSERIAL PRIMARY KEY,
        name                    TEXT UNIQUE NOT NULL,
        description             TEXT,
        representative_tickers  JSONB NOT NULL DEFAULT '[]',
        source                  TEXT NOT NULL DEFAULT 'manual',
        created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS instrument_dependencies (
        id            BIGSERIAL PRIMARY KEY,
        yf_symbol     TEXT NOT NULL,
        category_id   BIGINT NOT NULL REFERENCES dependency_categories(id) ON DELETE CASCADE,
        rationale     TEXT,
        source        TEXT NOT NULL DEFAULT 'manual',
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (yf_symbol, category_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_instrument_dependencies_symbol
        ON instrument_dependencies(yf_symbol)
    """,
]


def get_database_url() -> str:
    """Resolve the Postgres connection string from secrets or environment."""
    if st is not None:
        try:
            return st.secrets["postgres"]["url"]
        except Exception:
            pass

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No database URL found. Set it in .streamlit/secrets.toml under "
            "[postgres] url = ..., or export DATABASE_URL."
        )
    return url


_PG_NUMERIC_OID = 1700


def _connect():
    """Open a fresh connection with autocommit enabled."""
    conn = psycopg2.connect(get_database_url())
    conn.autocommit = True
    # By default psycopg2 returns NUMERIC columns as decimal.Decimal. Every
    # computation in this app (positions.py, portfolio configs, autosave
    # diffing) works in plain floats, and mixing float accumulators with
    # Decimal values raises a TypeError - so decode NUMERIC as float instead,
    # scoped to this connection only.
    float_numeric = psycopg2.extensions.new_type(
        (_PG_NUMERIC_OID,), "NUMERIC_AS_FLOAT",
        lambda value, cur: float(value) if value is not None else None,
    )
    psycopg2.extensions.register_type(float_numeric, conn)
    return conn


if st is not None:
    @st.cache_resource(show_spinner=False)
    def get_connection():
        """Return a cached connection, reconnecting if it has gone stale."""
        return _connect()
else:
    _cached_connection = None

    def get_connection():
        global _cached_connection
        if _cached_connection is None or _cached_connection.closed:
            _cached_connection = _connect()
        return _cached_connection


def _reconnect():
    """Force a fresh connection, used when a cached one has gone stale."""
    if st is not None:
        get_connection.clear()
        return get_connection()
    global _cached_connection
    _cached_connection = _connect()
    return _cached_connection


def execute(query: str, params: tuple = (), fetch: bool = False):
    """
    Run a query against the cached connection, transparently reconnecting
    once if the connection has gone stale (e.g. after the host auto-suspends
    or pauses the project during a quiet period).
    """
    for attempt in range(2):
        conn = get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                return None
        except psycopg2.OperationalError:
            if attempt == 0:
                logger.warning("Database connection stale, reconnecting...")
                _reconnect()
                continue
            raise


def execute_values(query: str, values: list, fetch: bool = False):
    """
    Like `execute`, but for psycopg2.extras.execute_values (bulk multi-row
    INSERT with a %s template). `query` must contain a single %s placeholder
    where the VALUES list goes.
    """
    if not values:
        return [] if fetch else None

    for attempt in range(2):
        conn = get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                result = psycopg2.extras.execute_values(cur, query, values, fetch=fetch)
                return result if fetch else None
        except psycopg2.OperationalError:
            if attempt == 0:
                logger.warning("Database connection stale, reconnecting...")
                _reconnect()
                continue
            raise


def ensure_schema() -> None:
    """Create tables/indexes if they don't already exist. Safe to call repeatedly."""
    for statement in _SCHEMA_STATEMENTS:
        execute(statement)
    logger.info("Database schema ensured.")


if st is not None:
    ensure_schema = st.cache_resource(show_spinner=False)(ensure_schema)
