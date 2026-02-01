# Portfolio Backtester

Interactive portfolio backtesting service that allows you to:
- Create portfolios with custom ticker weight distributions
- Backtest performance over different time periods (1d, 1w, 1m, 1y)
- See ROI percentage for each period
- Visualize portfolio performance with interactive charts
- Compare different weight distributions

## Features

- **Interactive Web Interface**: Streamlit-based UI with sliders to adjust portfolio weights
- **Concurrent Data Fetching**: Fetches historical prices for multiple tickers in parallel
- **Multiple Time Periods**: Analyze performance over 1 day, 1 week, 1 month, and 1 year
- **Real-time ROI Calculation**: See return percentages and absolute dollar returns
- **Visual Charts**: Interactive pie charts and time series charts using Plotly
- **Baseline Normalization**: Uses a configurable baseline amount ($10,000 default) for consistent comparisons

## Installation

Install required dependencies:

```bash
pip install streamlit plotly
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Web App

Run the Streamlit app:

```bash
streamlit run portfolio/app.py
```

The app will open in your browser. You can:
1. Enter tickers (comma-separated) in the sidebar
2. Adjust weight sliders for each ticker
3. Click "Calculate Performance" to see results
4. View charts showing portfolio value and returns over time

### Programmatic Usage

```python
from portfolio.portfolio_backtester import PortfolioBacktester

# Initialize backtester
backtester = PortfolioBacktester(baseline_amount=10000.0)

# Define portfolio
tickers = ['AAPL', 'MSFT', 'GOOGL']
weights = {'AAPL': 0.2, 'MSFT': 0.3, 'GOOGL': 0.5}

# Backtest
performance = backtester.backtest_portfolio(
    tickers=tickers,
    weights=weights,
    periods=['1d', '1w', '1m', '1y']
)

# Access results
for period, perf in performance.items():
    print(f"{period}: {perf.return_pct:.2f}% return")
    print(f"  ${perf.initial_value:.2f} → ${perf.final_value:.2f}")
```

### Test Script

Run the test script to verify everything works:

```bash
python3 portfolio/test_backtester.py
```

## Architecture

### Components

1. **PortfolioBacktester** (`portfolio_backtester.py`)
   - Main service class
   - Orchestrates backtesting workflow
   - Handles multiple time periods

2. **PriceDataFetcher** (`data_fetcher.py`)
   - Fetches historical price data from Finnhub API
   - Concurrent fetching for multiple tickers
   - Handles date lookups and price retrieval

3. **PortfolioCalculator** (`portfolio_calculator.py`)
   - Calculates initial shares based on weights
   - Computes portfolio values over time
   - Calculates returns and performance metrics

4. **Streamlit App** (`app.py`)
   - Interactive web interface
   - Real-time weight adjustment
   - Chart visualization

### Data Flow

```
1. User inputs tickers and weights
   ↓
2. PriceDataFetcher fetches historical prices (concurrent)
   ↓
3. PortfolioCalculator calculates initial shares
   ↓
4. PortfolioCalculator computes daily portfolio values
   ↓
5. PortfolioBacktester aggregates results by period
   ↓
6. Streamlit app displays charts and metrics
```

## How It Works

### Initial Setup

1. User defines tickers and target weights (e.g., AAPL 20%, MSFT 30%, GOOGL 50%)
2. System uses a baseline amount (default $10,000) for calculations
3. For each ticker:
   - Calculate dollar allocation: `weight × baseline_amount`
   - Get price on start date
   - Calculate shares: `dollar_allocation / start_price`

### Historical Calculation

For each day in the historical period:
1. Get current prices for all tickers
2. Calculate portfolio value: `sum(shares × current_price)`
3. Calculate return: `(current_value - initial_value) / initial_value × 100`
4. Store snapshot

### No Rebalancing

Shares remain constant throughout the period. Weights naturally drift as prices change.

## Example

**Input:**
- Tickers: AAPL, MSFT, GOOGL
- Weights: 20%, 30%, 50%
- Baseline: $10,000

**Initial Allocation:**
- AAPL: $2,000 → 13.33 shares (at $150/share)
- MSFT: $3,000 → 10.00 shares (at $300/share)
- GOOGL: $5,000 → 50.00 shares (at $100/share)

**After 1 week (prices changed):**
- AAPL: 13.33 shares × $155 = $2,066.67
- MSFT: 10.00 shares × $305 = $3,050.00
- GOOGL: 50.00 shares × $98 = $4,900.00
- **Total: $10,016.67 (0.17% return)**

## API Reference

### PortfolioBacktester

```python
backtest_portfolio(
    tickers: List[str],
    weights: Dict[str, float],
    periods: List[str] = ['1d', '1w', '1m', '1y']
) -> Dict[str, PortfolioPerformance]
```

### PortfolioPerformance

- `period`: Time period ('1d', '1w', '1m', '1y')
- `start_date`: Start date of period
- `end_date`: End date of period
- `initial_value`: Starting portfolio value
- `final_value`: Ending portfolio value
- `return_pct`: Percentage return
- `return_absolute`: Absolute dollar return
- `time_series`: List of daily snapshots

## Notes

- Requires Finnhub API key (set in `.env` file)
- Historical data availability depends on Finnhub API
- Free tier has rate limits (60 calls/minute)
- Missing data for specific dates will use closest available price
