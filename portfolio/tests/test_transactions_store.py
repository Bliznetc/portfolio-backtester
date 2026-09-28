#!/usr/bin/env python3
"""
Standalone round-trip check for transactions_store, including the
re-upload-is-a-no-op dedupe behavior.

Run with: python3 portfolio/tests/test_transactions_store.py
Requires DATABASE_URL env var (or .streamlit/secrets.toml) pointing at Postgres.
Creates and cleans up a throwaway user + portfolio config + transactions.
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.storage import db, portfolio_store, transactions_store


def _sample_rows():
    return [
        {
            "broker": "trading212",
            "broker_tx_id": "EOF_TEST_1",
            "action": "buy",
            "ticker": "NVDA",
            "isin": "US67066G1040",
            "executed_at": datetime(2025, 9, 16, 10, 50, 21, tzinfo=timezone.utc),
            "quantity": 1.0,
            "price": 177.17,
            "price_currency": "USD",
            "result": None,
            "result_currency": None,
            "total": 177.17,
            "total_currency": "USD",
            "raw": {"Action": "Market buy"},
        },
        {
            "broker": "trading212",
            "broker_tx_id": "h_derived_dividend_1",
            "action": "dividend",
            "ticker": "NVDA",
            "isin": "US67066G1040",
            "executed_at": datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc),
            "quantity": 1.0,
            "price": 0.01,
            "price_currency": "USD",
            "result": None,
            "result_currency": None,
            "total": 0.01,
            "total_currency": "USD",
            "raw": {"Action": "Dividend (Dividend)"},
        },
    ]


def main():
    db.ensure_schema()

    test_username = f"__test_{uuid.uuid4().hex[:8]}"
    print(f"Creating throwaway user '{test_username}'...")
    rows = db.execute(
        """
        INSERT INTO users (username, name, email, password_hash)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (test_username, "Test User", f"{test_username}@example.com", "not-a-real-hash"),
        fetch=True,
    )
    user_id = rows[0]["id"]

    try:
        config_id = portfolio_store.create_config(user_id, "T212 Test", [], {}, 0.0)

        print("Importing sample transactions...")
        sample = _sample_rows()
        inserted = transactions_store.upsert_transactions(config_id, sample)
        assert inserted == 2, f"expected 2 inserted, got {inserted}"
        print("✓ initial import inserted both rows")

        print("Re-importing the same rows (simulating an overlapping monthly export)...")
        inserted_again = transactions_store.upsert_transactions(config_id, sample)
        assert inserted_again == 0, f"expected 0 new inserts on re-upload, got {inserted_again}"
        stored = transactions_store.list_transactions(config_id)
        assert len(stored) == 2, f"expected still 2 rows stored, got {len(stored)}"
        print("✓ re-upload of the same rows was a no-op (dedup by broker_tx_id worked)")

        assert {s["ticker"] for s in stored} == {"NVDA"}
        assert {s["action"] for s in stored} == {"buy", "dividend"}
        print("✓ round-tripped fields look correct")

        print("Deleting transactions...")
        transactions_store.delete_transactions(config_id)
        assert transactions_store.list_transactions(config_id) == []
        print("✓ delete_transactions works")

        print("\nAll checks passed.")
    finally:
        print(f"Cleaning up throwaway user '{test_username}' (cascades to config + transactions)...")
        db.execute("DELETE FROM users WHERE id = %s", (user_id,))


if __name__ == "__main__":
    main()
