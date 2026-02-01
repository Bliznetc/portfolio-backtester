#!/usr/bin/env python3
"""
Interactive Streamlit app for portfolio backtesting.

Run with: streamlit run portfolio/app.py
"""

import sys
import logging
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio.portfolio_backtester import PortfolioBacktester
from portfolio.portfolio_calculator import PortfolioSnapshot
from portfolio.data_fetcher import PriceDataFetcher

# Page config
st.set_page_config(
    page_title="Portfolio Backtester",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'backtester' not in st.session_state:
    st.session_state.backtester = PortfolioBacktester(baseline_amount=10000.0)
if 'price_fetcher' not in st.session_state:
    st.session_state.price_fetcher = PriceDataFetcher()
if 'tickers' not in st.session_state:
    st.session_state.tickers = ['AAPL', 'MSFT', 'GOOGL']
if 'weights' not in st.session_state:
    st.session_state.weights = {ticker: 1.0 / len(st.session_state.tickers) 
                                for ticker in st.session_state.tickers}
if 'performance' not in st.session_state:
    st.session_state.performance = None
if 'slider_version' not in st.session_state:
    st.session_state.slider_version = 0

st.title("📊 Portfolio Backtester")
st.markdown("Build your pie and see how it would have performed!")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Baseline amount
    baseline = st.number_input(
        "Baseline Investment ($)",
        min_value=100.0,
        max_value=1000000.0,
        value=10000.0,
        step=100.0
    )
    st.session_state.backtester.baseline_amount = baseline
    
    # Ticker input
    st.subheader("Tickers")
    
    # Input field for adding new tickers
    new_ticker = st.text_input(
        "Enter a ticker to add",
        value="",
        key="new_ticker_input",
        placeholder="e.g., AAPL"
    )
    
    # Add ticker when user presses Enter or clicks button
    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("➕ Add Ticker", key="add_ticker_btn"):
            if new_ticker:
                ticker = new_ticker.strip().upper()
                if ticker and ticker not in st.session_state.tickers:
                    # Validate ticker using PriceDataFetcher
                    with st.spinner(f"Validating {ticker}..."):
                        if st.session_state.price_fetcher.validate_ticker(ticker):
                            st.session_state.tickers.append(ticker)
                            # Auto-rebalance to equal weights
                            st.session_state.weights = {
                                t: 1.0 / len(st.session_state.tickers) 
                                for t in st.session_state.tickers
                            }
                            st.session_state.performance = None
                            st.session_state.slider_version += 1
                            st.success(f"✅ {ticker} added successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Invalid ticker: {ticker}")
                            st.caption("Make sure the ticker exists and has recent price data.")
    
    with col_clear:
        if st.button("🗑️ Clear All", key="clear_all_btn"):
            st.session_state.tickers = []
            st.session_state.weights = {}
            st.session_state.performance = None
            st.rerun()
    
    st.markdown("---")
    st.markdown("**Current Tickers:**")
    
    # Display tickers with delete buttons
    if st.session_state.tickers:
        for i, ticker in enumerate(st.session_state.tickers):
            col_ticker, col_delete = st.columns([3, 1])
            with col_ticker:
                st.write(f"• {ticker}")
            with col_delete:
                if st.button("🗑️", key=f"delete_{ticker}_{i}", help=f"Remove {ticker}"):
                    st.session_state.tickers.remove(ticker)
                    # Auto-rebalance to equal weights
                    if st.session_state.tickers:
                        st.session_state.weights = {
                            t: 1.0 / len(st.session_state.tickers) 
                            for t in st.session_state.tickers
                        }
                    else:
                        st.session_state.weights = {}
                    st.session_state.performance = None
                    st.rerun()
    else:
        st.info("No tickers added yet. Enter a ticker above to get started.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Portfolio Allocation")
    
    # Weight sliders
    total_weight = 0.0
    weight_inputs = {}
    
    for ticker in st.session_state.tickers:
        current_weight = st.session_state.weights.get(ticker, 0.0) * 100
        weight = st.slider(
            f"{ticker} (%)",
            min_value=0.0,
            max_value=100.0,
            value=current_weight,
            step=0.1,
            key=f"weight_{ticker}_v{st.session_state.slider_version}"
        )
        weight_inputs[ticker] = weight / 100.0
        total_weight += weight
    
    # Display total weight with normalize button
    col_status, col_normalize = st.columns([2, 1])
    
    with col_status:
        if abs(total_weight - 100.0) > 0.1:
            st.warning(f"⚠️ Total weight: {total_weight:.1f}% (should be 100%)")
        else:
            st.success(f"✓ Total weight: {total_weight:.1f}%")
            st.session_state.weights = weight_inputs
    
    with col_normalize:
        if st.button("Normalize to 100%", width='stretch'):
            print(f"[DEBUG] Normalize button clicked")
            print(f"[DEBUG] Total weight (percent): {total_weight}")
            print(f"[DEBUG] Weight inputs (decimal): {weight_inputs}")
            print(f"[DEBUG] Tickers: {st.session_state.tickers}")
            
            if total_weight > 0 and weight_inputs:
                # weight_inputs are in decimal form (0-1), total_weight is in percent (0-100)
                # Convert total_weight to decimal for normalization
                total_weight_decimal = total_weight / 100.0
                new_weights = {
                    ticker: weight / total_weight_decimal 
                    for ticker, weight in weight_inputs.items()
                }
                print(f"[DEBUG] Total weight (decimal): {total_weight_decimal}")
                print(f"[DEBUG] New normalized weights: {new_weights}")
                
                # Update session state weights
                st.session_state.weights = new_weights
                
                # Increment slider version to force widget recreation
                st.session_state.slider_version += 1
                
                st.success(f"✅ Normalized to 100%!")
                st.rerun()
            elif not weight_inputs:
                st.error("❌ No tickers to normalize")
                print(f"[DEBUG] No weight inputs available")
            else:
                st.error(f"❌ Cannot normalize - total weight is {total_weight}")
                print(f"[DEBUG] Total weight is {total_weight}, cannot normalize")
    
    # Pie chart of allocation
    if st.session_state.weights:
        # Calculate dollar values for each ticker
        dollar_values = {
            ticker: weight * st.session_state.backtester.baseline_amount
            for ticker, weight in st.session_state.weights.items()
        }
        
        # Create pie chart with custom hover text showing dollar values
        fig_pie = px.pie(
            values=list(dollar_values.values()),
            names=list(dollar_values.keys()),
            title="Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Update traces to show label, dollar value, and percentage
        fig_pie.update_traces(
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Value: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
        )
        st.plotly_chart(fig_pie, width='content')

with col2:
    st.header("Performance Summary")
    
    if st.button("🔄 Calculate Performance", type="primary"):
        with st.spinner("Fetching data and calculating performance..."):
            try:
                st.session_state.performance = st.session_state.backtester.backtest_portfolio(
                    tickers=st.session_state.tickers,
                    weights=st.session_state.weights,
                    periods=['1d', '1w', '1m', '1y', '3y', '5y']
                )
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.performance:
        st.markdown("### Returns by Period")
        
        for period in ['1d', '1w', '1m', '1y', '3y', '5y']:
            if period in st.session_state.performance:
                perf = st.session_state.performance[period]
                
                # Color based on return
                color = "🟢" if perf.return_pct >= 0 else "🔴"
                
                st.metric(
                    label=f"{period.upper()} Return",
                    value=f"{perf.return_pct:.2f}%",
                    delta=f"${perf.return_absolute:.2f}",
                    delta_color="normal" if perf.return_pct >= 0 else "inverse"
                )
                
                st.caption(
                    f"${perf.initial_value:.2f} → ${perf.final_value:.2f}"
                )
        
        # Best and worst periods
        returns = {
            period: perf.return_pct 
            for period, perf in st.session_state.performance.items()
        }
        if returns:
            best_period = max(returns, key=returns.get)
            worst_period = min(returns, key=returns.get)
            
            st.markdown("---")
            st.markdown(f"**Best:** {best_period.upper()} ({returns[best_period]:.2f}%)")
            st.markdown(f"**Worst:** {worst_period.upper()} ({returns[worst_period]:.2f}%)")

# Performance charts
if st.session_state.performance:
    st.header("Performance Charts")
    
    # Time series chart
    period_tabs = st.tabs(['1D', '1W', '1M', '1Y', '3Y', '5Y'])
    
    for i, period in enumerate(['1d', '1w', '1m', '1y', '3y', '5y']):
        with period_tabs[i]:
            if period in st.session_state.performance:
                perf = st.session_state.performance[period]
                
                # Filter to only last trading day for 1D period
                time_series = perf.time_series
                if period == '1d' and time_series:
                    # Get the last date in the time series
                    last_date = time_series[-1].date.date()
                    # Filter to only snapshots from the last date
                    time_series = [s for s in time_series if s.date.date() == last_date]
                
                # Filter based on period type
                if period in ['1w', '1m', '1y', '3y', '5y']:
                    # For daily/longer data: filter to weekdays only (no specific hour filtering)
                    if time_series:
                        filtered_series = []
                        for s in time_series:
                            # Skip weekends (Monday=0, Sunday=6)
                            if s.date.weekday() >= 5:  # Saturday=5, Sunday=6
                                continue
                            filtered_series.append(s)
                        time_series = filtered_series
                
                # Create time series chart
                # (Timezone normalization now happens at data layer in portfolio_backtester.py)
                dates = [snapshot.date for snapshot in time_series]
                values = [snapshot.total_value for snapshot in time_series]
                returns = [snapshot.return_pct for snapshot in time_series]
                
                # Use sequential indices instead of datetime to avoid vertical lines across gaps
                indices = list(range(len(dates)))
                date_labels = [d.strftime('%Y-%m-%d %H:%M') for d in dates]
                
                # Portfolio value over time
                fig_value = go.Figure()
                fig_value.add_trace(go.Scatter(
                    x=indices,
                    y=values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=2),
                    text=date_labels,
                    hovertemplate='%{text}<br>Value: $%{y:,.2f}<extra></extra>'
                ))
                fig_value.add_hline(
                    y=st.session_state.backtester.baseline_amount,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Initial Value (${st.session_state.backtester.baseline_amount:,.0f})"
                )
                fig_value.update_layout(
                    title=f"Portfolio Value Over Time ({period.upper()})",
                    xaxis_title="Time",
                    yaxis_title="Value ($)",
                    hovermode='closest',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=indices[::max(1, len(indices)//10)],  # Show ~10 tick labels
                        ticktext=[date_labels[i] for i in indices[::max(1, len(indices)//10)]],
                        tickangle=-45
                    )
                )
                st.plotly_chart(fig_value, width='content')
                
                # Return percentage over time
                fig_return = go.Figure()
                fig_return.add_trace(go.Scatter(
                    x=indices,
                    y=returns,
                    mode='lines',
                    name='Return %',
                    line=dict(color='green' if perf.return_pct >= 0 else 'red', width=2),
                    fill='tozeroy',
                    text=date_labels,
                    hovertemplate='%{text}<br>Return: %{y:.2f}%<extra></extra>'
                ))
                fig_return.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray"
                )
                fig_return.update_layout(
                    title=f"Return Percentage Over Time ({period.upper()})",
                    xaxis_title="Time",
                    yaxis_title="Return (%)",
                    hovermode='closest',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=indices[::max(1, len(indices)//10)],
                        ticktext=[date_labels[i] for i in indices[::max(1, len(indices)//10)]],
                        tickangle=-45
                    )
                )
                st.plotly_chart(fig_return, width='content')
                
                # Individual ticker performance
                if perf.time_series:
                    st.subheader("Individual Ticker Performance")
                    
                    # Display tickers in rows dynamically: 5 columns per row
                    # This gives approximately ticker_count/5 rows
                    tickers_per_row = 5
                    for row_idx in range(0, len(st.session_state.tickers), tickers_per_row):
                        row_tickers = st.session_state.tickers[row_idx:row_idx + tickers_per_row]
                        ticker_cols = st.columns(len(row_tickers))
                        
                        for col_idx, ticker in enumerate(row_tickers):
                            with ticker_cols[col_idx]:
                                ticker_returns = [
                                    snapshot.individual_returns.get(ticker, 0.0)
                                    for snapshot in perf.time_series
                                ]
                                final_return = ticker_returns[-1] if ticker_returns else 0.0
                                
                                st.metric(
                                    label=ticker,
                                    value=f"{final_return:.2f}%"
                                )
            else:
                st.info(f"No data available for {period.upper()} period")

# Footer
st.markdown("---")
st.caption("💡 Adjust the sliders above and click 'Calculate Performance' to see how your portfolio would have performed!")
