"""
Session-state management for the "working" portfolio (tickers/weights/
baseline currently on screen) and its autosave to the database, plus a
couple of shared data-access helpers (current prices, cached transactions)
used by more than one page.
"""

from datetime import datetime

import streamlit as st

from portfolio.core import analyst_targets
from portfolio.core import classification as classification_core
from portfolio.core import dependencies as dependencies_core
from portfolio.core.data_fetcher import PriceDataFetcher
from portfolio.core.portfolio_backtester import PortfolioBacktester
from portfolio.core.symbol_resolver import resolve_yfinance_symbol
from portfolio.storage import classifications_store, dependencies_store, portfolio_store, transactions_store

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


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_price_target(symbol, price_scale):
    return analyst_targets.fetch_price_targets(symbol, price_scale)


def fetch_price_targets(computed_positions):
    """
    Best-effort analyst price target per ticker with quantity_held > 0, in
    each ticker's own price_currency (same GBX/GBP rescaling as
    fetch_current_prices). Cached for a day, since unlike a live price an
    analyst target doesn't move intraday - no reason to refetch it on every
    rerun the way the price is.
    """
    targets = {}
    for ticker, pos in computed_positions.items():
        if pos.quantity_held <= 0:
            continue
        symbol, price_scale = resolve_yfinance_symbol(ticker, pos.isin, pos.price_currency)
        targets[ticker] = _cached_price_target(symbol, price_scale)
    return targets


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_analyst_ratings(symbol, price_scale):
    return analyst_targets.fetch_analyst_ratings(symbol, price_scale)


def fetch_analyst_ratings(ticker, isin, price_currency):
    """Per-firm ratings/price targets for one ticker - see fetch_price_targets."""
    symbol, price_scale = resolve_yfinance_symbol(ticker, isin, price_currency)
    return _cached_analyst_ratings(symbol, price_scale)


@st.cache_data(ttl=3600, show_spinner=False)
def get_classification(ticker, isin, price_currency):
    """
    A ticker's classification row: yfinance's sector/industry/business
    summary, plus an AI-refined thematic tag once one has been generated
    (a no-op until an [anthropic] api_key is configured - see
    core.classification). Classified automatically the first time a
    ticker's detail page is opened, then persisted in the database - every
    call after that first one is a single read, not a re-fetch, and this
    cache just spares repeated DB round-trips within the same hour.
    """
    symbol, _ = resolve_yfinance_symbol(ticker, isin, price_currency)
    row = classifications_store.get_classification(symbol)

    if row is None:
        yf_data = classification_core.fetch_yf_classification(symbol)
        classifications_store.upsert_yf_layer(
            symbol, ticker, yf_data.sector, yf_data.industry, yf_data.business_summary
        )
        row = classifications_store.get_classification(symbol)

    if row and row.get("ai_classification") is None:
        ai_result = classification_core.classify_with_ai(
            ticker, row.get("yf_sector"), row.get("yf_industry"), row.get("business_summary")
        )
        if ai_result:
            classifications_store.upsert_ai_layer(symbol, *ai_result)
            row = classifications_store.get_classification(symbol)

    return row


@st.cache_data(ttl=3600, show_spinner=False)
def get_dependencies(ticker, isin, price_currency):
    """
    Every "what influences this instrument's price" category linked to this
    ticker - both AI-suggested and user-added manually, indistinguishable
    in the returned rows except for each row's "dependency_source"/
    "category_source" field. AI generation is retried on each call (cheap -
    a no-op instantly if no [gemini] api_key is configured) only while no
    AI-sourced link exists yet for this symbol, so: it runs once
    automatically the first time a ticker's detail page is opened, retries
    silently on every view until a key is added (also cheap and harmless),
    and stops retrying for good the first time it actually produces a
    suggestion - manual entries never affect this.
    """
    symbol, _ = resolve_yfinance_symbol(ticker, isin, price_currency)
    links = dependencies_store.list_dependencies_for_symbol(symbol)

    if not any(link["dependency_source"] == "ai" for link in links):
        classification_row = get_classification(ticker, isin, price_currency)
        if classification_row:
            existing_names = [c["name"] for c in dependencies_store.list_all_categories()]
            suggestions = dependencies_core.generate_dependencies_with_ai(
                ticker, classification_row.get("yf_sector"), classification_row.get("yf_industry"),
                classification_row.get("business_summary"), existing_categories=existing_names,
            )
            for s in suggestions:
                category_id = dependencies_store.get_or_create_category(
                    s.name, s.description, s.representative_tickers, source="ai"
                )
                dependencies_store.add_dependency(symbol, category_id, s.description, source="ai")
            if suggestions:
                links = dependencies_store.list_dependencies_for_symbol(symbol)
                list_all_categories.clear()

    return links


@st.cache_data(ttl=300, show_spinner=False)
def list_all_categories():
    """The shared category taxonomy, for a picker UI that links an
    instrument to an existing category instead of typing a new one."""
    return dependencies_store.list_all_categories()


def create_category_and_link(ticker, isin, price_currency, category_name, description, representative_tickers):
    """Creates a brand-new category (or reuses one with the same name, case-
    insensitively) and links this ticker to it."""
    symbol, _ = resolve_yfinance_symbol(ticker, isin, price_currency)
    category_id = dependencies_store.get_or_create_category(
        category_name, description, representative_tickers, source="manual"
    )
    dependencies_store.add_dependency(symbol, category_id, description, source="manual")
    get_dependencies.clear()
    list_all_categories.clear()


def link_existing_category(ticker, isin, price_currency, category_id, rationale=None):
    """Links this ticker to an already-existing category - the category's
    own representative tickers are untouched; edit those through
    add_ticker_to_category/remove_ticker_from_category instead."""
    symbol, _ = resolve_yfinance_symbol(ticker, isin, price_currency)
    dependencies_store.add_dependency(symbol, category_id, rationale, source="manual")
    get_dependencies.clear()


def remove_dependency(dependency_id):
    dependencies_store.remove_dependency(dependency_id)
    get_dependencies.clear()


def set_category_tickers(category_id, tickers):
    """Replaces a category's whole representative-tickers list - shared by
    every instrument linked to that category, not just the one whose page
    you're viewing."""
    normalized = sorted({t.strip().upper() for t in tickers if t.strip()})
    dependencies_store.set_category_tickers(category_id, normalized)
    get_dependencies.clear()
    list_all_categories.clear()


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
