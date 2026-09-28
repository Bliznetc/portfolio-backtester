"""
Storage for "what influences this instrument's price" dependency categories
and their links to instruments. Categories are global and reusable - the
same "Energy Costs" category can apply to many instruments - deduped by an
exact, case-insensitive name match. That's a deliberately simple rule: the
AI (or a user) phrasing the same idea differently ("Energy Costs" vs
"Energy Prices") will still create two separate categories - no fuzzy
matching is attempted here.
"""

import json
from typing import List, Optional

from . import db


def find_category_by_name(name: str) -> Optional[dict]:
    rows = db.execute(
        "SELECT * FROM dependency_categories WHERE lower(name) = lower(%s)",
        (name.strip(),),
        fetch=True,
    )
    return rows[0] if rows else None


def get_category(category_id: int) -> Optional[dict]:
    rows = db.execute(
        "SELECT * FROM dependency_categories WHERE id = %s", (category_id,), fetch=True
    )
    return rows[0] if rows else None


def list_all_categories() -> List[dict]:
    """Every category in the shared taxonomy, alphabetical - for a picker
    UI that lets a new instrument link to an existing category instead of
    creating a near-duplicate."""
    return db.execute("SELECT * FROM dependency_categories ORDER BY name", fetch=True)


def set_category_tickers(category_id: int, representative_tickers: List[str]) -> None:
    """
    Replaces a category's representative-tickers list wholesale - this is
    the one place tickers actually change, independent of any single
    instrument's link, so adding/removing a ticker here affects every
    instrument that shares this category.
    """
    db.execute(
        "UPDATE dependency_categories SET representative_tickers = %s WHERE id = %s",
        (json.dumps(representative_tickers), category_id),
    )


def create_category(
    name: str, description: Optional[str], representative_tickers: List[str], source: str,
) -> int:
    rows = db.execute(
        """
        INSERT INTO dependency_categories (name, description, representative_tickers, source)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
        """,
        (name.strip(), description, json.dumps(representative_tickers), source),
        fetch=True,
    )
    if rows:
        return rows[0]["id"]
    # Another call (or a case-different name) won the race/dedupe - reuse it.
    return find_category_by_name(name)["id"]


def get_or_create_category(
    name: str, description: Optional[str], representative_tickers: List[str], source: str,
) -> int:
    existing = find_category_by_name(name)
    if existing:
        return existing["id"]
    return create_category(name, description, representative_tickers, source)


def list_dependencies_for_symbol(yf_symbol: str) -> List[dict]:
    """Every category linked to this instrument, joined with the category's
    own details, oldest link first."""
    return db.execute(
        """
        SELECT
            d.id AS dependency_id, d.rationale, d.source AS dependency_source, d.created_at,
            c.id AS category_id, c.name, c.description,
            c.representative_tickers, c.source AS category_source
        FROM instrument_dependencies d
        JOIN dependency_categories c ON c.id = d.category_id
        WHERE d.yf_symbol = %s
        ORDER BY d.created_at
        """,
        (yf_symbol,),
        fetch=True,
    )


def add_dependency(yf_symbol: str, category_id: int, rationale: Optional[str], source: str) -> None:
    db.execute(
        """
        INSERT INTO instrument_dependencies (yf_symbol, category_id, rationale, source)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (yf_symbol, category_id) DO NOTHING
        """,
        (yf_symbol, category_id, rationale, source),
    )


def remove_dependency(dependency_id: int) -> None:
    db.execute("DELETE FROM instrument_dependencies WHERE id = %s", (dependency_id,))
