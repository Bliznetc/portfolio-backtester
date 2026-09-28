#!/usr/bin/env python3
"""
Interactive Streamlit app for portfolio backtesting.

Run with: streamlit run portfolio/app.py
"""

import sys
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio.core import fx, pnl, positions
from portfolio.core.data_fetcher import PriceDataFetcher
from portfolio.core.importers import revolut, trading212
from portfolio.core.portfolio_backtester import PortfolioBacktester
from portfolio.core.symbol_resolver import resolve_yfinance_symbol
from portfolio.storage import auth, db, portfolio_store, transactions_store
from portfolio.ui import charts

st.set_page_config(
    page_title="Portfolio Backtester",
    page_icon="📊",
    layout="wide"
)

PERIODS = ['1d', '1w', '1m', '1y', '3y', '5y']
DEFAULT_BASELINE = 10000.0
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL']
DEFAULT_PORTFOLIO_NAME = "My Portfolio"


# --- Working state and autosave -------------------------------------------

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


def _set_working_config(tickers, weights, baseline_amount):
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


def _load_config(config_row):
    cfg = config_row["config"]
    _set_working_config(
        cfg.get("tickers", []),
        cfg.get("weights", {}),
        cfg.get("baseline_amount", DEFAULT_BASELINE),
    )


def _autosave():
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


def _rebalance_equally():
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
    _load_config(configs[0])


# --- Login / signup -------------------------------------------------------

def render_signup_form():
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Username")
        name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")

    if not submitted:
        return
    if password != password_confirm:
        st.error("Passwords don't match.")
        return
    try:
        auth.sign_up(username, name, email, password)
    except ValueError as e:
        st.error(str(e))
        return
    st.success("Account created! Switch to the Log in tab to sign in.")


def require_login():
    """Show the login/signup page and halt the script until the user is authenticated."""
    try:
        db.ensure_schema()
    except Exception as e:
        st.error(f"Could not connect to the database: {e}")
        st.stop()

    authenticator = auth.get_authenticator()

    if st.session_state.get("authentication_status") is not True:
        st.title("📊 Portfolio Backtester")
        st.markdown("Build your pie and see how it would have performed!")

        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
        with login_tab:
            authenticator.login(location="main")
            if st.session_state.get("authentication_status") is False:
                st.error("Username or password is incorrect")
        with signup_tab:
            render_signup_form()
        st.stop()

    if "user_id" not in st.session_state:
        st.session_state.user_id = auth.get_user_id(st.session_state["username"])

    return authenticator


# --- Sidebar: saved portfolios --------------------------------------------

def render_portfolio_switcher():
    st.header("Your Portfolios")
    st.caption(f"Logged in as {st.session_state.get('name', st.session_state.get('username'))}")

    configs = portfolio_store.list_configs(st.session_state.user_id)
    names_by_id = {c["id"]: c["name"] for c in configs}

    if st.session_state.active_config_id not in names_by_id:
        # The active portfolio no longer exists (e.g. just deleted)
        st.session_state.active_config_id = configs[0]["id"]
        _load_config(configs[0])

    config_ids = list(names_by_id)
    selected_id = st.selectbox(
        "Active portfolio",
        options=config_ids,
        format_func=lambda config_id: names_by_id[config_id],
        index=config_ids.index(st.session_state.active_config_id),
        # Keyed on the active id so the widget re-initializes when the active
        # portfolio changes programmatically (new/deleted) instead of keeping stale state
        key=f"portfolio_selector_{st.session_state.active_config_id}",
    )
    if selected_id != st.session_state.active_config_id:
        st.session_state.active_config_id = selected_id
        _load_config(next(c for c in configs if c["id"] == selected_id))
        st.rerun()

    with st.expander("✏️ Rename"):
        render_rename_form(names_by_id[st.session_state.active_config_id])
    with st.expander("➕ New Portfolio"):
        render_new_portfolio_form()

    return configs


def render_rename_form(current_name):
    # Keyed on the active id so switching portfolios shows a fresh input
    # pre-filled with *that* portfolio's name, instead of the previous one's.
    with st.form(f"rename_form_{st.session_state.active_config_id}"):
        new_name = st.text_input("New name", value=current_name, label_visibility="collapsed")
        submitted = st.form_submit_button("Rename")

    if not submitted:
        return
    new_name = new_name.strip()
    if not new_name or new_name == current_name:
        return
    portfolio_store.update_config(st.session_state.active_config_id, name=new_name)
    st.rerun()


def render_new_portfolio_form():
    with st.form("new_portfolio_form", clear_on_submit=True):
        name = st.text_input("Name", placeholder="e.g. Revolut", label_visibility="collapsed")
        submitted = st.form_submit_button("➕ Create")

    if not submitted:
        return
    name = name.strip()
    if not name:
        st.error("Enter a name for the new portfolio.")
        return

    st.session_state.active_config_id = portfolio_store.create_config(
        st.session_state.user_id, name, [], {}, DEFAULT_BASELINE
    )
    _set_working_config([], {}, DEFAULT_BASELINE)
    st.success(f"Created '{name}'")
    st.rerun()


# --- Sidebar: danger zone (delete portfolio, log out) ----------------------

def render_danger_zone(authenticator, configs):
    confirm_key = f"confirm_delete_{st.session_state.active_config_id}"

    if len(configs) > 1:
        if st.session_state.get(confirm_key):
            current_name = next(
                (c["name"] for c in configs if c["id"] == st.session_state.active_config_id), "this portfolio"
            )
            st.warning(f"Delete '{current_name}'? This also deletes all its imported transactions - can't be undone.")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("Yes, delete it", type="primary", width='stretch'):
                    portfolio_store.delete_config(st.session_state.active_config_id)
                    del st.session_state.active_config_id
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", width='stretch'):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
        else:
            if st.button("🗑️ Delete Current Portfolio", type="primary", width='stretch'):
                st.session_state[confirm_key] = True
                st.rerun()

    authenticator.logout("Log out", location="sidebar")


# --- Sidebar: transaction import -------------------------------------------

IMPORTERS = {
    "Trading 212": trading212,
    "Revolut": revolut,
}


def _fetch_current_prices(computed_positions):
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


def _apply_transactions_to_working_config():
    """
    Recompute holdings from all imported transactions for the active
    portfolio and push the result into tickers/weights/baseline - both
    persisted and reflected in the working sliders.
    """
    all_txs = transactions_store.list_transactions(st.session_state.active_config_id)
    computed = positions.compute_positions(all_txs)
    current_prices, attempted_symbols = _fetch_current_prices(computed)

    weights, total_value, unpriced = positions.compute_weights(computed, current_prices)

    if weights:
        portfolio_store.update_config(
            st.session_state.active_config_id,
            tickers=list(weights.keys()),
            weights=weights,
            baseline_amount=total_value,
        )
        _set_working_config(list(weights.keys()), weights, total_value)

    if unpriced:
        details = ", ".join(f"{t} (tried {attempted_symbols.get(t, t)})" for t in sorted(unpriced))
        st.warning(
            f"Could not fetch a current price for: {details} - excluded from "
            "the pie chart, but still shown in the Positions table below."
        )


def render_transaction_import():
    st.subheader("Import Transactions")

    broker_name = st.selectbox("Broker", list(IMPORTERS.keys()), key="import_broker")
    uploader_version = st.session_state.get("import_uploader_version", 0)
    uploaded = st.file_uploader(
        "Upload export (CSV for Trading 212, PDF statement for Revolut)",
        type=["csv", "pdf"],
        key=f"import_uploader_v{uploader_version}"
    )

    if uploaded is None:
        return
    if not st.button("Import", key="import_btn"):
        return

    importer = IMPORTERS[broker_name]
    with st.spinner("Parsing and importing transactions..."):
        try:
            rows = importer.parse(uploaded)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            return
        inserted = transactions_store.upsert_transactions(st.session_state.active_config_id, rows)
        _apply_transactions_to_working_config()

    skipped_duplicates = len(rows) - inserted
    # st.success would render right before st.rerun() wipes the page, so it
    # never actually becomes visible - st.toast is built to survive exactly
    # this "act then rerun" pattern, showing on the next render instead.
    st.toast(
        f"Imported {inserted} new transaction(s) from {len(rows)} rows "
        f"({skipped_duplicates} already imported previously).",
        icon="✅",
    )
    st.session_state.import_uploader_version = uploader_version + 1
    st.rerun()


# --- Sidebar: baseline and tickers ----------------------------------------

def render_baseline_input():
    baseline = st.number_input(
        "Baseline Investment ($)",
        min_value=100.0,
        max_value=1000000.0,
        value=st.session_state.backtester.baseline_amount,
        step=100.0,
        key=f"baseline_v{st.session_state.slider_version}"
    )
    st.session_state.backtester.baseline_amount = baseline
    _autosave()


def add_ticker(raw_symbol):
    ticker = raw_symbol.strip().upper()
    if not ticker or ticker in st.session_state.tickers:
        return

    with st.spinner(f"Validating {ticker}..."):
        is_valid = st.session_state.price_fetcher.validate_ticker(ticker)

    if not is_valid:
        st.error(f"❌ Invalid ticker: {ticker}")
        st.caption("Make sure the ticker exists and has recent price data.")
        return

    st.session_state.tickers.append(ticker)
    _rebalance_equally()
    st.session_state.slider_version += 1
    _autosave()
    st.success(f"✅ {ticker} added successfully!")
    st.rerun()


def remove_ticker(ticker):
    st.session_state.tickers.remove(ticker)
    _rebalance_equally()
    _autosave()
    st.rerun()


def clear_tickers():
    st.session_state.tickers = []
    st.session_state.weights = {}
    st.session_state.performance = None
    _autosave()
    st.rerun()


def render_ticker_controls():
    st.subheader("Tickers")

    new_ticker = st.text_input(
        "Enter a ticker to add",
        value="",
        key="new_ticker_input",
        placeholder="e.g., AAPL"
    )

    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("➕ Add Ticker", key="add_ticker_btn") and new_ticker:
            add_ticker(new_ticker)
    with col_clear:
        if st.button("🗑️ Clear All", key="clear_all_btn"):
            clear_tickers()

    st.markdown("---")
    st.markdown("**Current Tickers:**")

    if not st.session_state.tickers:
        st.info("No tickers added yet. Enter a ticker above to get started.")
        return

    for i, ticker in enumerate(st.session_state.tickers):
        col_ticker, col_delete = st.columns([3, 1])
        with col_ticker:
            st.write(f"• {ticker}")
        with col_delete:
            if st.button("🗑️", key=f"delete_{ticker}_{i}", help=f"Remove {ticker}"):
                remove_ticker(ticker)


# --- Main area: allocation ------------------------------------------------

def normalize_weights(weight_inputs, total_weight):
    """Scale the slider weights so they sum to 100%."""
    if not weight_inputs:
        st.error("❌ No tickers to normalize")
        return
    if total_weight <= 0:
        st.error(f"❌ Cannot normalize - total weight is {total_weight}")
        return

    total_fraction = total_weight / 100.0
    st.session_state.weights = {
        ticker: weight / total_fraction for ticker, weight in weight_inputs.items()
    }
    _autosave()
    st.session_state.slider_version += 1
    st.success("✅ Normalized to 100%!")
    st.rerun()


def render_allocation():
    st.header("Portfolio Allocation")

    weight_inputs = {}
    total_weight = 0.0
    for ticker in st.session_state.tickers:
        percent = st.slider(
            f"{ticker} (%)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.weights.get(ticker, 0.0) * 100,
            step=0.1,
            key=f"weight_{ticker}_v{st.session_state.slider_version}"
        )
        weight_inputs[ticker] = percent / 100.0
        total_weight += percent

    col_status, col_normalize = st.columns([2, 1])
    with col_status:
        if abs(total_weight - 100.0) > 0.1:
            st.warning(f"⚠️ Total weight: {total_weight:.1f}% (should be 100%)")
        else:
            st.success(f"✓ Total weight: {total_weight:.1f}%")
            st.session_state.weights = weight_inputs
            _autosave()
    with col_normalize:
        if st.button("Normalize to 100%", width='stretch'):
            normalize_weights(weight_inputs, total_weight)

    if st.session_state.weights:
        fig = charts.allocation_pie(
            st.session_state.weights, st.session_state.backtester.baseline_amount
        )
        st.plotly_chart(fig, width='content')


# --- Main area: positions ---------------------------------------------------

def _format_currency_amounts(amounts_by_currency):
    if not amounts_by_currency:
        return "-"
    return " + ".join(
        f"{amount:.2f} {currency}" for currency, amount in sorted(amounts_by_currency.items())
    )


def _single_currency_amount(amounts_by_currency):
    """The lone value when a currency->amount dict has exactly one entry (the
    normal case for FIFO realized P/L, which is computed per-ticker in that
    ticker's own trading currency) - None otherwise, so the caller can fall
    back to text rather than silently mixing currencies into one number."""
    if amounts_by_currency and len(amounts_by_currency) == 1:
        return next(iter(amounts_by_currency.values()))
    return None


def _round_or_none(value, ndigits=2):
    return round(value, ndigits) if value is not None else None


_PNL_COLUMNS = ["Unrealized P/L (FIFO)", "Realized P/L (FIFO)"]


def _color_pnl(value):
    """Red for a loss, blue for a gain - used as a pandas Styler map function."""
    if not isinstance(value, (int, float)):
        return ""
    if value < 0:
        return "color: #ef4444"
    if value > 0:
        return "color: #3b82f6"
    return ""


def _styled(rows):
    """rows (list of dicts) -> a pandas Styler with P/L columns color-coded."""
    df = pd.DataFrame(rows)
    cols = [c for c in _PNL_COLUMNS if c in df.columns]
    return df.style.map(_color_pnl, subset=cols) if cols else df


@st.dialog("Transactions")
def _show_ticker_transactions_dialog(ticker, all_txs):
    st.subheader(ticker)
    ticker_txs = sorted(
        (t for t in all_txs if t["ticker"] == ticker), key=lambda t: t["executed_at"]
    )
    rows = [
        {
            "Date": t["executed_at"].strftime("%Y-%m-%d %H:%M"),
            "Broker": t["broker"],
            "Action": t["action"],
            "Quantity": t.get("quantity"),
            "Price": t.get("price"),
            "Currency": t.get("price_currency") or "-",
            "Total": t.get("total"),
            "Total Currency": t.get("total_currency") or "-",
        }
        for t in ticker_txs
    ]
    st.dataframe(rows, width='stretch', hide_index=True)


def render_positions_table():
    all_txs = transactions_store.list_transactions(st.session_state.active_config_id)
    if not all_txs:
        return

    computed = positions.compute_positions(all_txs)
    current_prices, _ = _fetch_current_prices(computed)
    fifo = pnl.compute_fifo_pnl(all_txs, current_prices)

    st.header("Positions")
    st.caption(
        "Computed from imported transactions for this portfolio. "
        "'Realized P/L (FIFO)' is our own oldest-lot-first calculation. "
        "Click a row to see that ticker's full transaction list."
    )

    all_rows = []
    cost_basis_by_ccy = defaultdict(float)
    unrealized_by_ccy = defaultdict(float)
    realized_fifo_by_ccy = defaultdict(float)
    dividends_by_ccy = defaultdict(float)
    total_transactions = 0

    for ticker in sorted(computed):
        pos = computed[ticker]
        fifo_pos = fifo.get(ticker)
        ccy = pos.price_currency
        cost_basis = fifo_pos.cost_basis_remaining if fifo_pos else None
        unrealized = fifo_pos.unrealized_pnl if fifo_pos else None
        realized_fifo = _single_currency_amount(fifo_pos.realized_pnl) if fifo_pos else None
        dividends_usd = fx.convert_amounts_to_usd(pos.dividends_received)

        all_rows.append({
            "Ticker": ticker,
            "Held": round(pos.quantity_held, 4),
            "Currency": ccy or "-",
            "Buy Volume": round(pos.total_bought_qty, 4),
            "Sell Volume": round(pos.total_sold_qty, 4),
            "Avg Buy Price (Held)": _round_or_none(fifo_pos.avg_cost_price if fifo_pos else None),
            "Cost Basis": _round_or_none(cost_basis),
            "Unrealized P/L (FIFO)": _round_or_none(unrealized),
            "Realized P/L (FIFO)": _round_or_none(realized_fifo),
            "Dividends (USD)": (
                _round_or_none(dividends_usd) if dividends_usd is not None
                else _format_currency_amounts(pos.dividends_received)
            ),
            "Transactions": pos.num_transactions,
        })

        total_transactions += pos.num_transactions
        if ccy:
            if cost_basis is not None:
                cost_basis_by_ccy[ccy] += cost_basis
            if unrealized is not None:
                unrealized_by_ccy[ccy] += unrealized
            if realized_fifo is not None:
                realized_fifo_by_ccy[ccy] += realized_fifo
        for div_ccy, amount in pos.dividends_received.items():
            dividends_by_ccy[div_ccy] += amount

    # One total row per currency, in its own (non-interactive) table above
    # the per-ticker one, so it's visible without scrolling and can never
    # get reordered by sorting the table below - st.dataframe has no
    # "pinned row" option, so a totals row living inside the sortable table
    # moves around with everything else the moment a column is sorted.
    # Held/Volume/Avg Buy Price aren't meaningful to sum across different
    # instruments (shares of AAPL plus shares of BP isn't a real quantity),
    # so those stay blank. Splitting by currency (rather than one combined
    # row) keeps every cell a genuine number rather than mixed-currency text.
    total_dividends_usd = fx.convert_amounts_to_usd(dividends_by_ccy)
    all_currencies = sorted(set(cost_basis_by_ccy) | set(unrealized_by_ccy) | set(realized_fifo_by_ccy))
    total_rows = [
        {
            "Ticker": f"TOTAL ({ccy})",
            "Held": None,
            "Currency": ccy,
            "Buy Volume": None,
            "Sell Volume": None,
            "Avg Buy Price (Held)": None,
            "Cost Basis": _round_or_none(cost_basis_by_ccy.get(ccy)),
            "Unrealized P/L (FIFO)": _round_or_none(unrealized_by_ccy.get(ccy)),
            "Realized P/L (FIFO)": _round_or_none(realized_fifo_by_ccy.get(ccy)),
            "Dividends (USD)": (
                _round_or_none(total_dividends_usd) if total_dividends_usd is not None
                else _format_currency_amounts(dividends_by_ccy)
            ),
            "Transactions": total_transactions,
        }
        for ccy in all_currencies
    ]

    row_height, header_height = 35, 38
    st.subheader("Totals")
    st.dataframe(
        _styled(total_rows), width='stretch', hide_index=True,
        height=header_height + row_height * len(total_rows),
    )

    st.subheader("By Ticker")
    open_only = st.checkbox("Show only open positions", value=True, key="positions_open_only")
    # The open-only toggle only affects which rows the table displays - the
    # totals above are always from every position regardless, since a closed
    # position's realized P/L and dividends are still real money.
    display_rows = [r for r in all_rows if not open_only or abs(r["Held"]) > 1e-6]

    event = st.dataframe(
        _styled(display_rows), width='stretch', hide_index=True,
        height=header_height + row_height * len(display_rows),
        on_select="rerun", selection_mode="single-row", key="positions_table",
    )
    selected = event.selection.rows if event and event.selection else []
    if selected:
        _show_ticker_transactions_dialog(display_rows[selected[0]]["Ticker"], all_txs)

    st.caption(
        "'Avg Buy Price (Held)' is the FIFO cost basis of shares still held "
        "only - not a lifetime average across shares already sold. It, Cost "
        "Basis, and the FIFO P/L columns are each in that row's own "
        "'Currency'. 'Dividends (USD)' is converted at today's exchange rate "
        "(a summary stat, unlike the trade-based columns, which stay "
        "unconverted) - falls back to unconverted text only if a currency's "
        "rate can't be fetched."
    )

    any_unknown_cost = any(fifo[t].quantity_unknown_cost > 1e-6 for t in fifo)
    if any_unknown_cost:
        st.caption(
            "⚠️ Some shares have no known cost basis (a corporate action like a "
            "merger/split, or a sale of shares bought before your earliest imported "
            "transaction) and are excluded from the FIFO P/L figures above."
        )


# --- Main area: performance -----------------------------------------------

def render_performance_summary():
    st.header("Performance Summary")

    if st.button("🔄 Calculate Performance", type="primary"):
        with st.spinner("Fetching data and calculating performance..."):
            try:
                st.session_state.performance = st.session_state.backtester.backtest_portfolio(
                    tickers=st.session_state.tickers,
                    weights=st.session_state.weights,
                    periods=PERIODS
                )
            except Exception as e:
                st.error(f"Error: {e}")

    performance = st.session_state.performance
    if not performance:
        return

    st.markdown("### Returns by Period")
    for period in PERIODS:
        if period not in performance:
            continue
        perf = performance[period]
        st.metric(
            label=f"{period.upper()} Return",
            value=f"{perf.return_pct:.2f}%",
            delta=f"${perf.return_absolute:.2f}",
            delta_color="normal" if perf.return_pct >= 0 else "inverse"
        )
        st.caption(f"${perf.initial_value:.2f} → ${perf.final_value:.2f}")

    returns = {period: perf.return_pct for period, perf in performance.items()}
    best_period = max(returns, key=returns.get)
    worst_period = min(returns, key=returns.get)
    st.markdown("---")
    st.markdown(f"**Best:** {best_period.upper()} ({returns[best_period]:.2f}%)")
    st.markdown(f"**Worst:** {worst_period.upper()} ({returns[worst_period]:.2f}%)")


def render_period_charts(period, perf):
    baseline = st.session_state.backtester.baseline_amount
    time_series = charts.prepare_time_series(period, perf.time_series)

    date_labels = [s.date.strftime('%Y-%m-%d %H:%M') for s in time_series]
    values = [s.total_value for s in time_series]
    returns = [s.return_pct for s in time_series]

    st.plotly_chart(
        charts.portfolio_value_chart(period, date_labels, values, baseline),
        width='content'
    )
    st.plotly_chart(
        charts.return_pct_chart(period, date_labels, returns, perf.return_pct),
        width='content'
    )

    if perf.time_series:
        render_individual_ticker_returns(perf)


def render_individual_ticker_returns(perf):
    st.subheader("Individual Ticker Performance")

    tickers = st.session_state.tickers
    tickers_per_row = 5
    for row_start in range(0, len(tickers), tickers_per_row):
        row_tickers = tickers[row_start:row_start + tickers_per_row]
        for col, ticker in zip(st.columns(len(row_tickers)), row_tickers):
            with col:
                final_return = perf.time_series[-1].individual_returns.get(ticker, 0.0)
                st.metric(label=ticker, value=f"{final_return:.2f}%")


def render_performance_charts():
    performance = st.session_state.performance
    if not performance:
        return

    st.header("Performance Charts")
    for tab, period in zip(st.tabs([p.upper() for p in PERIODS]), PERIODS):
        with tab:
            if period in performance:
                render_period_charts(period, performance[period])
            else:
                st.info(f"No data available for {period.upper()} period")


# --- Pages ------------------------------------------------------------------

def page_allocation():
    render_allocation()
    st.markdown("---")
    st.caption("💡 Adjust the sliders (or import transactions) to set your allocation.")


def page_performance():
    st.header("Configuration")
    render_baseline_input()
    render_ticker_controls()
    st.markdown("---")
    render_performance_summary()
    render_performance_charts()


def page_positions():
    render_positions_table()


# --- Page -----------------------------------------------------------------

def main():
    authenticator = require_login()
    init_session_state()
    load_active_config()

    st.title("📊 Portfolio Backtester")
    st.markdown("Build your pie and see how it would have performed!")

    with st.sidebar:
        configs = render_portfolio_switcher()
        st.markdown("---")
        render_transaction_import()
        st.markdown("---")
        render_danger_zone(authenticator, configs)

    pages = st.navigation([
        st.Page(page_allocation, title="Allocation", icon="🥧", url_path="allocation", default=True),
        st.Page(page_performance, title="Performance", icon="📈", url_path="performance"),
        st.Page(page_positions, title="Positions", icon="📋", url_path="positions"),
    ])
    pages.run()


main()
