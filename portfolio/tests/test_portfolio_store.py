#!/usr/bin/env python3
"""
Standalone round-trip check for portfolio_store CRUD.

Run with: python3 portfolio/test_portfolio_store.py
Requires DATABASE_URL env var (or .streamlit/secrets.toml) pointing at Postgres.
Creates and cleans up a throwaway user + config row.
"""

import sys
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.storage import db, portfolio_store


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
        print("Creating config...")
        config_id = portfolio_store.create_config(
            user_id, "Test Portfolio", ["AAPL", "MSFT"], {"AAPL": 0.6, "MSFT": 0.4}, 5000.0
        )

        print("Listing configs...")
        configs = portfolio_store.list_configs(user_id)
        assert len(configs) == 1, f"expected 1 config, got {len(configs)}"
        assert configs[0]["id"] == config_id
        print("✓ list_configs OK")

        print("Getting config...")
        fetched = portfolio_store.get_config(config_id)
        assert fetched["config"]["tickers"] == ["AAPL", "MSFT"]
        assert fetched["config"]["weights"] == {"AAPL": 0.6, "MSFT": 0.4}
        assert fetched["config"]["baseline_amount"] == 5000.0
        print("✓ get_config OK")

        print("Updating config (partial update)...")
        portfolio_store.update_config(config_id, weights={"AAPL": 0.5, "MSFT": 0.5})
        updated = portfolio_store.get_config(config_id)
        assert updated["config"]["weights"] == {"AAPL": 0.5, "MSFT": 0.5}
        assert updated["config"]["tickers"] == ["AAPL", "MSFT"], "tickers should be untouched by partial update"
        print("✓ update_config OK (partial update preserved other fields)")

        print("Deleting config...")
        portfolio_store.delete_config(config_id)
        assert portfolio_store.list_configs(user_id) == []
        print("✓ delete_config OK")

        print("\nAll checks passed.")
    finally:
        print(f"Cleaning up throwaway user '{test_username}'...")
        db.execute("DELETE FROM users WHERE id = %s", (user_id,))


if __name__ == "__main__":
    main()
