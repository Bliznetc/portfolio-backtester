#!/usr/bin/env python3
"""
Unit tests for PortfolioStore.

Runs against a temporary SQLite database by default. Set TEST_DATABASE_URL to run
the same tests against Postgres (all tables in that database are dropped afterwards).
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio.storage import PortfolioStore, Transaction, metadata


class TestPortfolioStore(unittest.TestCase):
    """Test suite for saved portfolios and transactions."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{self.tmp_dir.name}/test.db"
        self.store = PortfolioStore(url)

    def tearDown(self):
        metadata.drop_all(self.store.engine)
        self.store.engine.dispose()
        self.tmp_dir.cleanup()

    def _tx(self, tx_id, ticker="AAPL", tx_type="buy", quantity=1.0, price=100.0):
        return Transaction(
            broker_tx_id=tx_id, date=datetime(2026, 1, 5), type=tx_type,
            ticker=ticker, quantity=quantity, price=price, currency="USD",
            total=quantity * price,
        )

    def test_save_and_get_portfolio(self):
        portfolio_id = self.store.save_portfolio("Tech", {"AAPL": 0.4, "MSFT": 0.6})

        portfolio = self.store.get_portfolio(portfolio_id)
        self.assertEqual(portfolio.name, "Tech")
        self.assertEqual(portfolio.broker, "manual")
        self.assertEqual(portfolio.weights, {"AAPL": 0.4, "MSFT": 0.6})

    def test_save_existing_name_replaces_weights(self):
        first_id = self.store.save_portfolio("Tech", {"AAPL": 0.4, "MSFT": 0.6})
        second_id = self.store.save_portfolio("Tech", {"NVDA": 1.0})

        self.assertEqual(first_id, second_id)
        self.assertEqual(self.store.get_portfolio(first_id).weights, {"NVDA": 1.0})
        self.assertEqual(len(self.store.list_portfolios()), 1)

    def test_save_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.store.save_portfolio("   ", {"AAPL": 1.0})

    def test_list_portfolios_is_per_user_and_sorted(self):
        self.store.save_portfolio("Zeta", {"AAPL": 1.0})
        self.store.save_portfolio("Alpha", {"MSFT": 1.0})
        self.store.save_portfolio("Other user", {"MSFT": 1.0}, user_id="someone-else")

        names = [p.name for p in self.store.list_portfolios()]
        self.assertEqual(names, ["Alpha", "Zeta"])

    def test_get_portfolio_by_name(self):
        self.store.save_portfolio("Revolut", {"AAPL": 1.0}, broker="revolut")

        portfolio = self.store.get_portfolio_by_name("Revolut")
        self.assertEqual(portfolio.broker, "revolut")
        self.assertIsNone(self.store.get_portfolio_by_name("Missing"))

    def test_update_weights(self):
        portfolio_id = self.store.save_portfolio("Tech", {"AAPL": 1.0})
        self.store.update_weights(portfolio_id, {"AAPL": 0.5, "GOOGL": 0.5})

        self.assertEqual(
            self.store.get_portfolio(portfolio_id).weights, {"AAPL": 0.5, "GOOGL": 0.5}
        )

    def test_delete_portfolio_removes_weights_and_transactions(self):
        portfolio_id = self.store.save_portfolio("Tech", {"AAPL": 1.0})
        self.store.add_transactions(portfolio_id, [self._tx("t1")])

        self.store.delete_portfolio(portfolio_id)

        self.assertIsNone(self.store.get_portfolio(portfolio_id))
        self.assertEqual(self.store.list_transactions(portfolio_id), [])

    def test_add_transactions_skips_duplicates(self):
        portfolio_id = self.store.save_portfolio("T212", {}, broker="trading212")

        inserted, skipped = self.store.add_transactions(
            portfolio_id, [self._tx("t1"), self._tx("t2"), self._tx("t2")]
        )
        self.assertEqual((inserted, skipped), (2, 1))

        # Re-uploading an overlapping export only adds the new row
        inserted, skipped = self.store.add_transactions(
            portfolio_id, [self._tx("t1"), self._tx("t2"), self._tx("t3")]
        )
        self.assertEqual((inserted, skipped), (1, 2))
        self.assertEqual(len(self.store.list_transactions(portfolio_id)), 3)

    def test_transactions_are_separated_per_portfolio(self):
        t212_id = self.store.save_portfolio("T212", {}, broker="trading212")
        revolut_id = self.store.save_portfolio("Revolut", {}, broker="revolut")

        # Same broker id in two portfolios is not a duplicate
        self.store.add_transactions(t212_id, [self._tx("same-id", ticker="AAPL")])
        self.store.add_transactions(revolut_id, [self._tx("same-id", ticker="TSLA")])

        self.assertEqual([t.ticker for t in self.store.list_transactions(t212_id)], ["AAPL"])
        self.assertEqual([t.ticker for t in self.store.list_transactions(revolut_id)], ["TSLA"])

    def test_list_transactions_filter_by_ticker(self):
        portfolio_id = self.store.save_portfolio("T212", {}, broker="trading212")
        self.store.add_transactions(portfolio_id, [
            self._tx("t1", ticker="AAPL"),
            self._tx("t2", ticker="MSFT"),
            self._tx("t3", ticker="AAPL", tx_type="sell"),
        ])

        aapl = self.store.list_transactions(portfolio_id, ticker="AAPL")
        self.assertEqual([t.type for t in aapl], ["buy", "sell"])

    def test_transaction_round_trip(self):
        portfolio_id = self.store.save_portfolio("T212", {}, broker="trading212")
        original = self._tx("t1", quantity=2.5, price=180.25)

        self.store.add_transactions(portfolio_id, [original])

        self.assertEqual(self.store.list_transactions(portfolio_id), [original])


if __name__ == '__main__':
    unittest.main()
