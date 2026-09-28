"""
Sidebar sections: saved-portfolio management (switch/rename/create/delete,
logout) and broker transaction import.
"""

import streamlit as st

from portfolio.core import positions
from portfolio.core.importers import revolut, trading212
from portfolio.storage import portfolio_store, transactions_store
from portfolio.ui.working_config import (
    DEFAULT_BASELINE, cached_transactions, fetch_current_prices, load_config, set_working_config,
)


# --- Saved portfolios --------------------------------------------

def render_portfolio_switcher():
    st.header("Your Portfolios")
    st.caption(f"Logged in as {st.session_state.get('name', st.session_state.get('username'))}")

    configs = portfolio_store.list_configs(st.session_state.user_id)
    names_by_id = {c["id"]: c["name"] for c in configs}

    if st.session_state.active_config_id not in names_by_id:
        # The active portfolio no longer exists (e.g. just deleted)
        st.session_state.active_config_id = configs[0]["id"]
        load_config(configs[0])

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
        load_config(next(c for c in configs if c["id"] == selected_id))
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
    set_working_config([], {}, DEFAULT_BASELINE)
    st.success(f"Created '{name}'")
    st.rerun()


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


# --- Transaction import -------------------------------------------

IMPORTERS = {
    "Trading 212": trading212,
    "Revolut": revolut,
}


def _apply_transactions_to_working_config():
    """
    Recompute holdings from all imported transactions for the active
    portfolio and push the result into tickers/weights/baseline - both
    persisted and reflected in the working sliders.
    """
    all_txs = cached_transactions(st.session_state.active_config_id)
    computed = positions.compute_positions(all_txs)
    current_prices, attempted_symbols = fetch_current_prices(computed)

    weights, total_value, unpriced = positions.compute_weights(computed, current_prices)

    if weights:
        portfolio_store.update_config(
            st.session_state.active_config_id,
            tickers=list(weights.keys()),
            weights=weights,
            baseline_amount=total_value,
        )
        set_working_config(list(weights.keys()), weights, total_value)

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
        cached_transactions.clear()
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
