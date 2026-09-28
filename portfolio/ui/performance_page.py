"""
"Performance Summary" (Calculate Performance button + return metrics) and
"Performance Charts" sections of the Performance page.
"""

import streamlit as st

from portfolio.ui import charts

PERIODS = ['1d', '1w', '1m', '1y', '3y', '5y']


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
