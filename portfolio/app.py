#!/usr/bin/env python3
"""
Interactive Streamlit app for portfolio backtesting.

Run with: streamlit run portfolio/app.py
"""

import sys
import logging
from pathlib import Path

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio.ui.allocation_page import render_allocation, render_baseline_input, render_ticker_controls
from portfolio.ui.login_page import require_login
from portfolio.ui.performance_page import render_performance_charts, render_performance_summary
from portfolio.ui.positions_page import (
    POSITIONS_PATH, activate_portfolio_from_query, render_positions_table, render_ticker_transactions_page,
)
from portfolio.ui.sidebar import render_danger_zone, render_portfolio_switcher, render_transaction_import
from portfolio.ui.working_config import init_session_state, load_active_config

st.set_page_config(
    page_title="Portfolio Backtester",
    page_icon="📊",
    layout="wide"
)

PERFORMANCE_PATH = "performance"


# --- Pages ------------------------------------------------------------------

def page_performance():
    st.header("Configuration")
    render_baseline_input()
    render_ticker_controls()
    st.markdown("---")
    render_allocation()
    st.caption("💡 Adjust the sliders (or import transactions) to set your allocation.")
    st.markdown("---")
    render_performance_summary()
    render_performance_charts()


def page_positions():
    # The ticker-detail case is intercepted in main() before page routing
    # runs at all (see the comment there) - reached only for the table view.
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

    positions_page = st.Page(page_positions, title="Positions", icon="📋", url_path=POSITIONS_PATH, default=True)
    pages = st.navigation([
        positions_page,
        st.Page(page_performance, title="Performance", icon="📈", url_path=PERFORMANCE_PATH),
    ])

    # A hard navigation (a real <a href> link, a bookmark, a typed URL)
    # starts a brand-new session, and on that first run st.navigation just
    # renders whichever page has default=True - it doesn't reliably honor
    # the browser's actual requested path for function-based pages. So the
    # ticker-detail link from the Positions table is handled here, before
    # handing off to page routing, rather than depending on landing on the
    # Positions page via url_path matching.
    ticker = st.query_params.get("ticker")
    if ticker:
        activate_portfolio_from_query(st.query_params.get("portfolio"))
        render_ticker_transactions_page(ticker, positions_page)
    else:
        pages.run()


# Streamlit runs the entry script as "__main__", so this guard keeps normal
# app behavior while letting tests import this module without executing the
# whole app as a side effect of the import.
if __name__ == "__main__":
    main()
