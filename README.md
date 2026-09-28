# Portfolio Backtester

Interactive portfolio backtesting application built with Streamlit.

🔗 **Live app:** [banger-tester.streamlit.app](https://banger-tester.streamlit.app/)

## Features
- User accounts (sign up / log in) with saved portfolio configurations
- Switch between multiple saved portfolios from the sidebar, with autosave
- Import broker transaction history (Trading 212 CSV export) and see real
  positions, average buy/sell price, and realized profit/dividends per stock
- Multi-ticker portfolio allocation
- Historical performance backtesting (1D to 5Y)
- Interactive charts with Plotly
- Support for multi-exchange portfolios

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
```

This app stores user accounts and saved portfolios in Postgres (a free
hosted instance - [Neon](https://neon.tech) or [Supabase](https://supabase.com)
both work). Create a `.streamlit/secrets.toml` file (already gitignored) with:

```toml
[postgres]
url = "postgresql://<user>:<password>@<host>/<dbname>?sslmode=require"

[auth]
cookie_name = "portfolio_app_auth"
cookie_key = "<fixed random 32+ char secret — never regenerate>"
cookie_expiry_days = 30
```

Then run:

```bash
streamlit run portfolio/app.py
```

Sign up for an account on first run — the schema (users, portfolio_configs
tables) is created automatically.

### Deploying

The hosted app (Streamlit Community Cloud) needs the same `[postgres]` and
`[auth]` values set in the app's **Settings → Secrets** panel.
