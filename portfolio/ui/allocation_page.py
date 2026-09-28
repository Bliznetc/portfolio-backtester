"""
"Configuration" (baseline + ticker list) and "Portfolio Allocation" (weight
sliders + pie chart) sections of the Performance page.
"""

import streamlit as st

from portfolio.ui import charts
from portfolio.ui.working_config import autosave, rebalance_equally


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
    autosave()


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
    rebalance_equally()
    st.session_state.slider_version += 1
    autosave()
    st.success(f"✅ {ticker} added successfully!")
    st.rerun()


def remove_ticker(ticker):
    st.session_state.tickers.remove(ticker)
    rebalance_equally()
    autosave()
    st.rerun()


def clear_tickers():
    st.session_state.tickers = []
    st.session_state.weights = {}
    st.session_state.performance = None
    autosave()
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
    autosave()
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
            autosave()
    with col_normalize:
        if st.button("Normalize to 100%", width='stretch'):
            normalize_weights(weight_inputs, total_weight)

    if st.session_state.weights:
        fig = charts.allocation_pie(
            st.session_state.weights, st.session_state.backtester.baseline_amount
        )
        st.plotly_chart(fig, width='content')
