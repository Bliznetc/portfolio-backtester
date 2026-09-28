"""
Session-state management for the "working" portfolio (tickers/weights/
baseline currently on screen) and its autosave to the database, plus a
couple of shared data-access helpers (current prices, cached transactions)
used by more than one page.
"""

from datetime import datetime

import streamlit as st

from portfolio.core.data_fetcher import PriceDataFetcher
from portfolio.core.portfolio_backtester import PortfolioBacktester
from portfolio.core.symbol_resolver import resolve_yfinance_symbol
from portfolio.storage import portfolio_store, transactions_store

DEFAULT_BASELINE = 10000.0
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL']
DEFAULT_PORTFOLIO_NAME = "My Portfolio"


def _snapshot():
    """
    Current tickers/weights/baseline as a hashable tuple, for autosave diffing.
    Weights are rounded so slider float round-trips don't count as edits.
    """
    return (
        tuple(st.session_state.tickers),
        tuple(sorted((t, round(w, 6)) for t, w in st.session_state.weights.items())),
        st.session_state.backtester.baseline_amount,
    )


def set_working_config(tickers, weights, baseline_amount):
    """
    Replace the working portfolio (used when switching or creating one).
    Bumping slider_version gives the sliders/number input new widget keys,
    so they re-render with the new values instead of their stale ones.
    """
    st.session_state.tickers = tickers
    st.session_state.weights = weights
    st.session_state.backtester.baseline_amount = baseline_amount
    st.session_state.slider_version += 1
    st.session_state.performance = None
    st.session_state.last_saved_snapshot = _snapshot()


def load_config(config_row):
    cfg = config_row["config"]
    set_working_config(
        cfg.get("tickers", []),
        cfg.get("weights", {}),
        cfg.get("baseline_amount", DEFAULT_BASELINE),
    )


def autosave():
    """Persist the working config if it changed since the last save."""
    snapshot = _snapshot()
    if snapshot == st.session_state.get("last_saved_snapshot"):
        return
    portfolio_store.update_config(
        st.session_state.active_config_id,
        tickers=st.session_state.tickers,
        weights=st.session_state.weights,
        baseline_amount=st.session_state.backtester.baseline_amount,
    )
    st.session_state.last_saved_snapshot = snapshot


def rebalance_equally():
    st.session_state.weights = st.session_state.backtester.get_equal_weights(
        st.session_state.tickers
    )
    st.session_state.performance = None


def init_session_state():
    if 'backtester' not in st.session_state:
        st.session_state.backtester = PortfolioBacktester(baseline_amount=DEFAULT_BASELINE)
    if 'price_fetcher' not in st.session_state:
        st.session_state.price_fetcher = PriceDataFetcher()
    if 'performance' not in st.session_state:
        st.session_state.performance = None
    if 'slider_version' not in st.session_state:
        st.session_state.slider_version = 0


def load_active_config():
    """Ensure the user has a saved portfolio and one is loaded as the working config."""
    if "active_config_id" in st.session_state:
        return

    user_id = st.session_state.user_id
    configs = portfolio_store.list_configs(user_id)
    if not configs:
        portfolio_store.create_config(
            user_id,
            DEFAULT_PORTFOLIO_NAME,
            DEFAULT_TICKERS,
            st.session_state.backtester.get_equal_weights(DEFAULT_TICKERS),
            DEFAULT_BASELINE,
        )
        configs = portfolio_store.list_configs(user_id)

    st.session_state.active_config_id = configs[0]["id"]
    load_config(configs[0])


def fetch_current_prices(computed_positions):
    """
    Best-effort current price per ticker with quantity_held > 0, in each
    ticker's own price_currency (see symbol_resolver for the GBX/GBP
    rescaling). Returns (current_prices, attempted_symbols) - the latter
    is only for building a helpful "couldn't price X" message.
    """
    current_prices = {}
    attempted_symbols = {}
    for ticker, pos in computed_positions.items():
        if pos.quantity_held <= 0:
            continue
        symbol, price_scale = resolve_yfinance_symbol(ticker, pos.isin, pos.price_currency)
        attempted_symbols[ticker] = symbol
        price = st.session_state.price_fetcher.get_price_on_date(symbol, datetime.now())
        if price is not None:
            current_prices[ticker] = price * price_scale
    return current_prices, attempted_symbols


@st.cache_data(ttl=300, show_spinner=False)
def cached_transactions(config_id):
    """
    The whole portfolio's transaction list, cached so switching to a ticker's
    detail view (or toggling the open-only checkbox, or anything else that
    reruns the script) doesn't re-hit the database or redo FIFO/positions
    math every time. Cleared explicitly right after a new import, since
    that's the only thing that changes the underlying data.
    """
    return transactions_store.list_transactions(config_id)
