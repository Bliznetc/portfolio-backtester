#!/usr/bin/env python3
"""
Standalone check that the DB connection and schema setup work.

Run with: python3 portfolio/test_db.py
Requires DATABASE_URL env var (or .streamlit/secrets.toml) pointing at Postgres.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.storage import db


def main():
    print("Connecting to database...")
    db.ensure_schema()
    print("✓ Schema ensured (users, portfolio_configs)")

    tables = db.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name IN ('users', 'portfolio_configs')
        """,
        fetch=True,
    )
    table_names = {row["table_name"] for row in tables}
    assert "users" in table_names, "users table missing"
    assert "portfolio_configs" in table_names, "portfolio_configs table missing"
    print(f"✓ Confirmed tables exist: {sorted(table_names)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
