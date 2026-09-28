#!/usr/bin/env python3
"""
Standalone round-trip check for classifications_store.

Run with: python3 portfolio/test_classifications_store.py
Requires DATABASE_URL env var (or .streamlit/secrets.toml) pointing at Postgres.
Creates and cleans up a throwaway classification row.
"""

import sys
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.storage import classifications_store, db


def main():
    db.ensure_schema()

    test_symbol = f"__TEST_{uuid.uuid4().hex[:8]}"

    try:
        print("Missing row returns None...")
        assert classifications_store.get_classification(test_symbol) is None
        print("✓ get_classification OK (missing row)")

        print("Inserting yfinance layer...")
        classifications_store.upsert_yf_layer(
            test_symbol, "TEST", "Technology", "Data Center Providers", "A fake business summary."
        )
        row = classifications_store.get_classification(test_symbol)
        assert row["yf_sector"] == "Technology"
        assert row["yf_industry"] == "Data Center Providers"
        assert row["business_summary"] == "A fake business summary."
        assert row["ai_classification"] is None
        print("✓ upsert_yf_layer OK")

        print("Re-inserting yfinance layer updates in place (no duplicate row)...")
        classifications_store.upsert_yf_layer(
            test_symbol, "TEST", "Technology", "Updated Industry", "Updated summary."
        )
        row = classifications_store.get_classification(test_symbol)
        assert row["yf_industry"] == "Updated Industry"
        print("✓ upsert_yf_layer OK (ON CONFLICT update)")

        print("Adding AI layer...")
        classifications_store.upsert_ai_layer(
            test_symbol, "AI/HPC Data Center Infrastructure", "Builds and operates GPU data centers.",
            "claude-haiku-4-5-20251001",
        )
        row = classifications_store.get_classification(test_symbol)
        assert row["ai_classification"] == "AI/HPC Data Center Infrastructure"
        assert row["ai_rationale"] == "Builds and operates GPU data centers."
        assert row["ai_model"] == "claude-haiku-4-5-20251001"
        assert row["ai_generated_at"] is not None
        # The yfinance layer set by the earlier upsert must survive the AI-layer update.
        assert row["yf_industry"] == "Updated Industry"
        print("✓ upsert_ai_layer OK")

        print("\nAll checks passed.")
    finally:
        print(f"Cleaning up throwaway row '{test_symbol}'...")
        db.execute("DELETE FROM instrument_classifications WHERE yf_symbol = %s", (test_symbol,))


if __name__ == "__main__":
    main()
