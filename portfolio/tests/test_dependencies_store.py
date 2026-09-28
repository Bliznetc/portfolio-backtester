#!/usr/bin/env python3
"""
Standalone round-trip check for dependencies_store.

Run with: python3 portfolio/test_dependencies_store.py
Requires DATABASE_URL env var (or .streamlit/secrets.toml) pointing at Postgres.
Creates and cleans up throwaway categories/links.
"""

import sys
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.storage import dependencies_store, db


def main():
    db.ensure_schema()

    test_category_name = f"__Test Category {uuid.uuid4().hex[:8]}"
    test_symbol = f"__TEST_{uuid.uuid4().hex[:8]}"
    category_id = None

    try:
        print("Missing category returns None...")
        assert dependencies_store.find_category_by_name(test_category_name) is None
        print("✓ find_category_by_name OK (missing)")

        print("Creating category...")
        category_id = dependencies_store.create_category(
            test_category_name, "A fake dependency category.", ["XLE", "VDE"], "manual"
        )
        assert category_id is not None
        print("✓ create_category OK")

        print("Case-insensitive lookup finds it...")
        found = dependencies_store.find_category_by_name(test_category_name.upper())
        assert found is not None and found["id"] == category_id
        print("✓ find_category_by_name OK (case-insensitive)")

        print("get_or_create_category reuses the existing row...")
        reused_id = dependencies_store.get_or_create_category(
            test_category_name, "different description", [], "ai"
        )
        assert reused_id == category_id, "should reuse the existing category, not create a duplicate"
        print("✓ get_or_create_category OK (no duplicate)")

        print("Linking category to a test instrument...")
        dependencies_store.add_dependency(test_symbol, category_id, "Test rationale.", "manual")
        links = dependencies_store.list_dependencies_for_symbol(test_symbol)
        assert len(links) == 1, f"expected 1 link, got {len(links)}"
        assert links[0]["name"] == test_category_name
        assert links[0]["representative_tickers"] == ["XLE", "VDE"]
        assert links[0]["dependency_source"] == "manual"
        print("✓ add_dependency / list_dependencies_for_symbol OK")

        print("Re-adding the same link is a no-op (ON CONFLICT DO NOTHING)...")
        dependencies_store.add_dependency(test_symbol, category_id, "Different rationale.", "ai")
        links = dependencies_store.list_dependencies_for_symbol(test_symbol)
        assert len(links) == 1, "duplicate link should not have been inserted"
        print("✓ add_dependency OK (no duplicate link)")

        print("Removing the link...")
        dependencies_store.remove_dependency(links[0]["dependency_id"])
        assert dependencies_store.list_dependencies_for_symbol(test_symbol) == []
        print("✓ remove_dependency OK")

        print("\nAll checks passed.")
    finally:
        print("Cleaning up...")
        db.execute("DELETE FROM instrument_dependencies WHERE yf_symbol = %s", (test_symbol,))
        if category_id is not None:
            db.execute("DELETE FROM dependency_categories WHERE id = %s", (category_id,))


if __name__ == "__main__":
    main()
