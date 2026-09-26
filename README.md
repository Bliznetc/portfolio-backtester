# Portfolio Backtester

Interactive portfolio backtesting application built with Streamlit.

🔗 **Live app:** [banger-tester.streamlit.app](https://banger-tester.streamlit.app/)

## Features
- Multi-ticker portfolio allocation
- Historical performance backtesting (1D to 5Y)
- Interactive charts with Plotly
- Support for multi-exchange portfolios
- Save and load named portfolios (Supabase/Postgres or local SQLite)

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run portfolio/app.py
```

Price data comes from Yahoo Finance via `yfinance` - no API key needed.

### Saved portfolios (database)
By default portfolios are saved to a local SQLite file (`data/portfolios.db`) - no setup needed.

To use Postgres (e.g. Supabase) instead, set `DATABASE_URL`:
- **Streamlit Cloud:** app Settings -> Secrets
- **Locally:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored)

```toml
DATABASE_URL = "postgresql://postgres.<project-ref>:<password>@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"
```

Use Supabase's **Session pooler** connection string (Streamlit Cloud can't reach the direct IPv6 host).
Tables are created automatically on first start.

### Tests
```bash
python -m pytest portfolio
# Run storage tests against Postgres too:
TEST_DATABASE_URL=postgresql://... python -m pytest portfolio/test_storage.py
```
