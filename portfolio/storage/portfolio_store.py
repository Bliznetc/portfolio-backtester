"""
CRUD for saved portfolio configurations (tickers + weights + baseline amount),
stored one-per-row as JSONB in the portfolio_configs table.
"""

import json
import logging
from typing import Dict, List, Optional

from . import db

logger = logging.getLogger(__name__)


def list_configs(user_id: int) -> List[dict]:
    """List a user's saved configs, most recently updated first."""
    rows = db.execute(
        """
        SELECT id, name, config, updated_at
        FROM portfolio_configs
        WHERE user_id = %s
        ORDER BY updated_at DESC
        """,
        (user_id,),
        fetch=True,
    )
    return rows


def get_config(config_id: int) -> Optional[dict]:
    """Fetch a single config by id."""
    rows = db.execute(
        """
        SELECT id, user_id, name, config, updated_at
        FROM portfolio_configs
        WHERE id = %s
        """,
        (config_id,),
        fetch=True,
    )
    return rows[0] if rows else None


def create_config(
    user_id: int,
    name: str,
    tickers: List[str],
    weights: Dict[str, float],
    baseline_amount: float,
) -> int:
    """Create a new saved config and return its id."""
    config = {"tickers": tickers, "weights": weights, "baseline_amount": baseline_amount}
    rows = db.execute(
        """
        INSERT INTO portfolio_configs (user_id, name, config)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (user_id, name, json.dumps(config)),
        fetch=True,
    )
    config_id = rows[0]["id"]
    logger.info(f"Created portfolio config {config_id} ('{name}') for user {user_id}")
    return config_id


def update_config(
    config_id: int,
    *,
    tickers: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    baseline_amount: Optional[float] = None,
    name: Optional[str] = None,
) -> None:
    """
    Partially update a saved config. Only fields that are passed (not None)
    are changed; the rest of the JSONB config is left as-is. Always bumps
    updated_at.
    """
    existing = get_config(config_id)
    if existing is None:
        raise ValueError(f"No portfolio config with id {config_id}")

    current = existing["config"]
    if tickers is not None:
        current["tickers"] = tickers
    if weights is not None:
        current["weights"] = weights
    if baseline_amount is not None:
        current["baseline_amount"] = baseline_amount

    if name is not None:
        db.execute(
            """
            UPDATE portfolio_configs
            SET name = %s, config = %s, updated_at = now()
            WHERE id = %s
            """,
            (name, json.dumps(current), config_id),
        )
    else:
        db.execute(
            """
            UPDATE portfolio_configs
            SET config = %s, updated_at = now()
            WHERE id = %s
            """,
            (json.dumps(current), config_id),
        )


def delete_config(config_id: int) -> None:
    """Delete a saved config."""
    db.execute("DELETE FROM portfolio_configs WHERE id = %s", (config_id,))
    logger.info(f"Deleted portfolio config {config_id}")
