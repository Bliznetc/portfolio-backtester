"""
Positions page: the Totals/By Ticker tables (holdings, FIFO cost basis,
realized/unrealized P&L, dividends) and the per-ticker transaction-detail
view reached from the "Details" link column.
"""

from collections import defaultdict
from urllib.parse import quote

import pandas as pd
import streamlit as st

from portfolio.core import fx, pnl, positions
from portfolio.storage import portfolio_store
from portfolio.ui.working_config import cached_transactions, fetch_current_prices, load_config

POSITIONS_PATH = "positions"

# st.dataframe height (px) needed to fit a given number of rows with no
# internal scrollbar.
_DATAFRAME_ROW_HEIGHT = 35
_DATAFRAME_HEADER_HEIGHT = 38


def _dataframe_height(num_rows):
    return _DATAFRAME_HEADER_HEIGHT + _DATAFRAME_ROW_HEIGHT * num_rows


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
    """Red for a loss, green for a gain - used as a pandas Styler map function."""
    if not isinstance(value, (int, float)):
        return ""
    if value < 0:
        return "color: #ef4444"
    if value > 0:
        return "color: #22c55e"
    return ""


def _styled(rows):
    """rows (list of dicts) -> a pandas Styler with P/L columns color-coded."""
    df = pd.DataFrame(rows)
    cols = [c for c in _PNL_COLUMNS if c in df.columns]
    return df.style.map(_color_pnl, subset=cols) if cols else df


def activate_portfolio_from_query(portfolio_param):
    """
    Honor a ?portfolio=<id> link so a shared/bookmarked ticker URL opens the
    right portfolio in a fresh session. Looked up through the current user's
    own configs, so a hand-edited id can't pull up someone else's portfolio.
    """
    if not portfolio_param or not portfolio_param.isdigit():
        return
    wanted = int(portfolio_param)
    if wanted == st.session_state.get("active_config_id"):
        return
    configs = portfolio_store.list_configs(st.session_state.user_id)
    match = next((c for c in configs if c["id"] == wanted), None)
    if match:
        st.session_state.active_config_id = wanted
        load_config(match)


def render_ticker_transactions_page(ticker, positions_page):
    if st.button("← Back to Positions"):
        # st.switch_page needs the actual Page object this session's
        # st.navigation registered it as (not just its url_path) - and,
        # since we bypassed pages.run() while showing this ticker's detail,
        # st.navigation never got a chance to mark Positions as the
        # "current" page for this session, so a plain st.rerun() here would
        # fall back to whichever page has default=True instead.
        st.switch_page(positions_page)

    st.header(f"Transactions: {ticker}")

    all_txs = cached_transactions(st.session_state.active_config_id)
    ticker_txs = sorted(
        (t for t in all_txs if t["ticker"] == ticker), key=lambda t: t["executed_at"]
    )
    if not ticker_txs:
        st.info(f"No transactions found for {ticker}.")
        return

    computed = positions.compute_positions(ticker_txs)
    pos = computed.get(ticker)
    current_prices, _ = fetch_current_prices(computed)
    fifo_pos = pnl.compute_fifo_pnl(ticker_txs, current_prices).get(ticker)
    ccy = (pos.price_currency if pos else None) or ""
    current_price = current_prices.get(ticker)

    def _fmt(value, currency=ccy):
        return f"{value:.2f} {currency}" if value is not None else "-"

    cols = st.columns(5)
    cols[0].metric("Held", f"{pos.quantity_held:.4f}" if pos else "-")
    cols[1].metric("Avg Buy Price", _fmt(fifo_pos.avg_cost_price if fifo_pos else None))
    cols[2].metric("Current Price", _fmt(current_price))
    cols[3].metric("Cost Basis", _fmt(fifo_pos.cost_basis_remaining if fifo_pos else None))
    unrealized = fifo_pos.unrealized_pnl if fifo_pos else None
    cols[4].metric(
        "Unrealized P/L",
        _fmt(unrealized),
        delta=round(unrealized, 2) if unrealized is not None else None,
    )
    st.markdown("---")

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
    st.dataframe(rows, width='stretch', hide_index=True, height=_dataframe_height(len(rows)))


def render_positions_table():
    config_id = st.session_state.active_config_id
    all_txs = cached_transactions(config_id)
    if not all_txs:
        return

    computed = positions.compute_positions(all_txs)
    current_prices, _ = fetch_current_prices(computed)
    fifo = pnl.compute_fifo_pnl(all_txs, current_prices)

    st.header("Positions")
    st.caption(
        "Computed from imported transactions for this portfolio. "
        "'Realized P/L (FIFO)' is our own oldest-lot-first calculation. "
        "Click a row's → to see that ticker's full transaction list."
    )

    all_rows = []
    cost_basis_by_ccy = defaultdict(float)
    unrealized_by_ccy = defaultdict(float)
    realized_fifo_by_ccy = defaultdict(float)
    dividends_by_ccy = defaultdict(float)

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
            "Avg Buy Price (Held)": _round_or_none(fifo_pos.avg_cost_price if fifo_pos else None),
            "Cost Basis": _round_or_none(cost_basis),
            "Unrealized P/L (FIFO)": _round_or_none(unrealized),
            "Realized P/L (FIFO)": _round_or_none(realized_fifo),
            "Buy Volume": round(pos.total_bought_qty, 4),
            "Sell Volume": round(pos.total_sold_qty, 4),
            "Dividends (USD)": (
                _round_or_none(dividends_usd) if dividends_usd is not None
                else _format_currency_amounts(pos.dividends_received)
            ),
            "Currency": ccy or "-",
            # Absolute path, not a bare "?ticker=..." - Streamlit sets
            # <base href="/"> in the page head, so a relative query-only link
            # resolves against the app root and loses the page path. The
            # portfolio id rides along so the link still opens the right
            # portfolio when followed in a fresh session (a plain <a href> is
            # a full page load, which resets st.session_state).
            "Details": f"/{POSITIONS_PATH}?ticker={quote(ticker)}&portfolio={config_id}",
        })

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
            "Avg Buy Price (Held)": None,
            "Cost Basis": _round_or_none(cost_basis_by_ccy.get(ccy)),
            "Unrealized P/L (FIFO)": _round_or_none(unrealized_by_ccy.get(ccy)),
            "Realized P/L (FIFO)": _round_or_none(realized_fifo_by_ccy.get(ccy)),
            "Buy Volume": None,
            "Sell Volume": None,
            "Dividends (USD)": (
                _round_or_none(total_dividends_usd) if total_dividends_usd is not None
                else _format_currency_amounts(dividends_by_ccy)
            ),
            "Currency": ccy,
        }
        for ccy in all_currencies
    ]

    st.subheader("Totals")
    st.dataframe(
        _styled(total_rows), width='stretch', hide_index=True,
        height=_dataframe_height(len(total_rows)),
    )

    st.subheader("By Ticker")
    open_only = st.checkbox("Show only open positions", value=True, key="positions_open_only")
    # The open-only toggle only affects which rows the table displays - the
    # totals above are always from every position regardless, since a closed
    # position's realized P/L and dividends are still real money.
    display_rows = [r for r in all_rows if not open_only or abs(r["Held"]) > 1e-6]

    st.dataframe(
        _styled(display_rows), width='stretch', hide_index=True,
        height=_dataframe_height(len(display_rows)),
        column_config={
            "Details": st.column_config.LinkColumn("Details", display_text="→", width="small"),
        },
    )

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
