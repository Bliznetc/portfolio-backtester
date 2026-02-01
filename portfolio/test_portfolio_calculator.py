"""
Unit tests for portfolio_calculator module using unittest framework.
"""

import unittest
from datetime import datetime, timedelta
from .portfolio_calculator import PortfolioCalculator, PortfolioSnapshot


class TestPortfolioCalculator(unittest.TestCase):
    """Test cases for PortfolioCalculator class."""
    
    def setUp(self):
        """Create a PortfolioCalculator instance for each test."""
        self.calculator = PortfolioCalculator()
    
    # Tests for calculate_initial_shares
    
    def test_calculate_initial_shares_equal_weights(self):
        """Test initial share calculation with equal weights."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        weights = {'AAPL': 0.333, 'MSFT': 0.333, 'GOOGL': 0.334}
        initial_value = 10000.0
        start_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 120.0}
        
        shares = self.calculator.calculate_initial_shares(
            tickers=tickers,
            weights=weights,
            initial_value=initial_value,
            start_prices=start_prices
        )
        
        self.assertEqual(len(shares), 3)
        # Each ticker should get roughly 1/3 of initial value
        self.assertAlmostEqual(shares['AAPL'], 10000 * 0.333 / 150.0, places=1)
        self.assertAlmostEqual(shares['MSFT'], 10000 * 0.333 / 300.0, places=1)
        self.assertAlmostEqual(shares['GOOGL'], 10000 * 0.334 / 120.0, places=1)
    
    def test_calculate_initial_shares_custom_weights(self):
        """Test initial share calculation with custom weights."""
        tickers = ['AAPL', 'MSFT']
        weights = {'AAPL': 0.6, 'MSFT': 0.4}
        initial_value = 10000.0
        start_prices = {'AAPL': 150.0, 'MSFT': 300.0}
        
        shares = self.calculator.calculate_initial_shares(
            tickers=tickers,
            weights=weights,
            initial_value=initial_value,
            start_prices=start_prices
        )
        
        # AAPL should get 60% of portfolio
        self.assertAlmostEqual(shares['AAPL'], 6000.0 / 150.0, places=1)
        # MSFT should get 40% of portfolio
        self.assertAlmostEqual(shares['MSFT'], 4000.0 / 300.0, places=1)
    
    def test_calculate_initial_shares_missing_price(self):
        """Test that missing start prices raise ValueError."""
        tickers = ['AAPL', 'MSFT']
        weights = {'AAPL': 0.5, 'MSFT': 0.5}
        initial_value = 10000.0
        start_prices = {'AAPL': 150.0}  # MSFT price is missing
        
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate_initial_shares(
                tickers=tickers,
                weights=weights,
                initial_value=initial_value,
                start_prices=start_prices
            )
        
        self.assertIn('Missing start price for MSFT', str(context.exception))
    
    def test_calculate_initial_shares_normalize_weights(self):
        """Test weight normalization when weights don't sum to 1.0."""
        tickers = ['AAPL', 'MSFT']
        weights = {'AAPL': 0.4, 'MSFT': 0.4}  # Sum to 0.8, not 1.0
        initial_value = 10000.0
        start_prices = {'AAPL': 150.0, 'MSFT': 300.0}
        
        shares = self.calculator.calculate_initial_shares(
            tickers=tickers,
            weights=weights,
            initial_value=initial_value,
            start_prices=start_prices
        )
        
        # Shares should be calculated based on normalized weights
        self.assertGreater(shares['AAPL'], 0)
        self.assertGreater(shares['MSFT'], 0)
    
    def test_calculate_initial_shares_invalid_price(self):
        """Test that invalid (zero or negative) prices raise ValueError."""
        tickers = ['AAPL', 'MSFT']
        weights = {'AAPL': 0.5, 'MSFT': 0.5}
        initial_value = 10000.0
        start_prices = {'AAPL': 150.0, 'MSFT': 0.0}  # Invalid price for MSFT
        
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate_initial_shares(
                tickers=tickers,
                weights=weights,
                initial_value=initial_value,
                start_prices=start_prices
            )
        
        self.assertIn('Invalid price', str(context.exception))
    
    # Tests for calculate_portfolio_value
    
    def test_calculate_portfolio_value_basic(self):
        """Test basic portfolio value calculation."""
        shares = {'AAPL': 10.0, 'MSFT': 5.0, 'GOOGL': 20.0}
        prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 120.0}
        
        total_value, individual_values = self.calculator.calculate_portfolio_value(shares, prices)
        
        expected_total = (10.0 * 150.0) + (5.0 * 300.0) + (20.0 * 120.0)
        self.assertAlmostEqual(total_value, expected_total, places=2)
        self.assertAlmostEqual(individual_values['AAPL'], 1500.0, places=2)
        self.assertAlmostEqual(individual_values['MSFT'], 1500.0, places=2)
        self.assertAlmostEqual(individual_values['GOOGL'], 2400.0, places=2)
    
    def test_calculate_portfolio_value_missing_price(self):
        """Test portfolio value calculation with missing prices."""
        shares = {'AAPL': 10.0, 'MSFT': 5.0}
        prices = {'AAPL': 150.0}  # MSFT price is missing
        
        total_value, individual_values = self.calculator.calculate_portfolio_value(shares, prices)
        
        self.assertAlmostEqual(total_value, 1500.0, places=2)
        self.assertAlmostEqual(individual_values['AAPL'], 1500.0, places=2)
        self.assertEqual(individual_values['MSFT'], 0.0)
    
    def test_calculate_portfolio_value_zero_shares(self):
        """Test portfolio value calculation with zero shares."""
        shares = {'AAPL': 0.0, 'MSFT': 5.0}
        prices = {'AAPL': 150.0, 'MSFT': 300.0}
        
        total_value, individual_values = self.calculator.calculate_portfolio_value(shares, prices)
        
        self.assertAlmostEqual(total_value, 1500.0, places=2)
        self.assertEqual(individual_values['AAPL'], 0.0)
        self.assertAlmostEqual(individual_values['MSFT'], 1500.0, places=2)
    
    # Tests for calculate_returns
    
    def test_calculate_returns_positive(self):
        """Test return calculation for positive returns."""
        current_value = 12000.0
        initial_value = 10000.0
        individual_values = {'AAPL': 1800.0, 'MSFT': 1800.0, 'GOOGL': 2400.0}
        initial_individual_values = {'AAPL': 1500.0, 'MSFT': 1500.0, 'GOOGL': 2400.0}
        
        return_pct, individual_returns = self.calculator.calculate_returns(
            current_value=current_value,
            initial_value=initial_value,
            individual_values=individual_values,
            initial_individual_values=initial_individual_values
        )
        
        self.assertAlmostEqual(return_pct, 20.0, places=1)  # 2000 / 10000 * 100
        self.assertAlmostEqual(individual_returns['AAPL'], 20.0, places=1)
        self.assertAlmostEqual(individual_returns['MSFT'], 20.0, places=1)
        self.assertAlmostEqual(individual_returns['GOOGL'], 0.0, places=1)
    
    def test_calculate_returns_negative(self):
        """Test return calculation for negative returns."""
        current_value = 8000.0
        initial_value = 10000.0
        individual_values = {'AAPL': 1200.0, 'MSFT': 1200.0}
        initial_individual_values = {'AAPL': 1500.0, 'MSFT': 1500.0}
        
        return_pct, individual_returns = self.calculator.calculate_returns(
            current_value=current_value,
            initial_value=initial_value,
            individual_values=individual_values,
            initial_individual_values=initial_individual_values
        )
        
        self.assertAlmostEqual(return_pct, -20.0, places=1)  # -2000 / 10000 * 100
        self.assertAlmostEqual(individual_returns['AAPL'], -20.0, places=1)
        self.assertAlmostEqual(individual_returns['MSFT'], -20.0, places=1)
    
    def test_calculate_returns_zero_initial_value(self):
        """Test return calculation with zero initial value."""
        current_value = 100.0
        initial_value = 0.0
        individual_values = {'AAPL': 100.0}
        initial_individual_values = {'AAPL': 0.0}
        
        return_pct, individual_returns = self.calculator.calculate_returns(
            current_value=current_value,
            initial_value=initial_value,
            individual_values=individual_values,
            initial_individual_values=initial_individual_values
        )
        
        self.assertEqual(return_pct, 0.0)
        self.assertEqual(individual_returns, {})  # Empty dict when initial_value is 0
    
    def test_calculate_returns_zero_initial_individual_value(self):
        """Test individual return calculation with zero initial individual value."""
        current_value = 100.0
        initial_value = 100.0
        individual_values = {'AAPL': 100.0}
        initial_individual_values = {'AAPL': 0.0}  # No initial value for AAPL
        
        return_pct, individual_returns = self.calculator.calculate_returns(
            current_value=current_value,
            initial_value=initial_value,
            individual_values=individual_values,
            initial_individual_values=initial_individual_values
        )
        
        self.assertEqual(individual_returns['AAPL'], 0.0)
    
    # Tests for calculate_historical_performance
    
    def test_calculate_historical_performance_basic(self):
        """Test historical performance calculation with simple data."""
        shares = {'AAPL': 10.0}
        
        # Create mock price data with 3 data points
        now = datetime.now()
        price_data = {
            'AAPL': {
                'dates': [now - timedelta(days=2), now - timedelta(days=1), now],
                'prices': [150.0, 155.0, 160.0],
                'opens': [149.0, 154.0, 159.0],
                'highs': [151.0, 156.0, 161.0],
                'lows': [149.0, 154.0, 159.0],
                'volumes': [1000000, 1100000, 1200000]
            }
        }
        
        initial_value = 1500.0
        start_date = now - timedelta(days=2)
        
        snapshots = self.calculator.calculate_historical_performance(
            shares=shares,
            price_data=price_data,
            initial_value=initial_value,
            start_date=start_date
        )
        
        self.assertEqual(len(snapshots), 3)
        self.assertAlmostEqual(snapshots[0].total_value, 1500.0, places=2)
        self.assertAlmostEqual(snapshots[1].total_value, 1550.0, places=2)
        self.assertAlmostEqual(snapshots[2].total_value, 1600.0, places=2)
    
    def test_calculate_historical_performance_empty_data(self):
        """Test historical performance with empty price data."""
        shares = {'AAPL': 10.0}
        price_data = {}
        
        snapshots = self.calculator.calculate_historical_performance(
            shares=shares,
            price_data=price_data,
            initial_value=1500.0,
            start_date=datetime.now()
        )
        
        self.assertEqual(len(snapshots), 0)
    
    def test_calculate_historical_performance_snapshot_structure(self):
        """Test that snapshots have correct structure."""
        shares = {'AAPL': 10.0}
        now = datetime.now()
        
        price_data = {
            'AAPL': {
                'dates': [now],
                'prices': [150.0],
                'opens': [149.0],
                'highs': [151.0],
                'lows': [149.0],
                'volumes': [1000000]
            }
        }
        
        snapshots = self.calculator.calculate_historical_performance(
            shares=shares,
            price_data=price_data,
            initial_value=1500.0,
            start_date=now
        )
        
        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        
        self.assertIsInstance(snapshot, PortfolioSnapshot)
        self.assertEqual(snapshot.date, now)
        self.assertGreater(snapshot.total_value, 0)
        self.assertIn('AAPL', snapshot.individual_values)
        self.assertIn('AAPL', snapshot.individual_returns)
    
    def test_calculate_historical_performance_multiple_tickers(self):
        """Test historical performance with multiple tickers."""
        shares = {'AAPL': 10.0, 'MSFT': 5.0}
        now = datetime.now()
        
        price_data = {
            'AAPL': {
                'dates': [now],
                'prices': [150.0],
                'opens': [149.0],
                'highs': [151.0],
                'lows': [149.0],
                'volumes': [1000000]
            },
            'MSFT': {
                'dates': [now],
                'prices': [300.0],
                'opens': [299.0],
                'highs': [301.0],
                'lows': [299.0],
                'volumes': [800000]
            }
        }
        
        snapshots = self.calculator.calculate_historical_performance(
            shares=shares,
            price_data=price_data,
            initial_value=3000.0,
            start_date=now
        )
        
        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        
        # Total should be (10 * 150) + (5 * 300) = 3000
        self.assertAlmostEqual(snapshot.total_value, 3000.0, places=2)
        self.assertAlmostEqual(snapshot.individual_values['AAPL'], 1500.0, places=2)
        self.assertAlmostEqual(snapshot.individual_values['MSFT'], 1500.0, places=2)
    
    def test_calculate_historical_performance_changing_prices(self):
        """Test historical performance across multiple dates with changing prices."""
        # Portfolio with 3 tickers
        shares = {'AAPL': 100.0, 'MSFT': 50.0, 'GOOGL': 25.0}
        
        # Create 5 dates with different prices for each ticker
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(5)]
        
        # Price progression for each ticker
        price_data = {
            'AAPL': {
                'dates': dates,
                'prices': [150.0, 155.0, 152.0, 158.0, 165.0],
                'opens': [149.0, 154.0, 151.0, 157.0, 164.0],
                'highs': [151.0, 156.0, 153.0, 159.0, 166.0],
                'lows': [149.0, 154.0, 151.0, 157.0, 164.0],
                'volumes': [1000000] * 5
            },
            'MSFT': {
                'dates': dates,
                'prices': [300.0, 305.0, 310.0, 308.0, 315.0],
                'opens': [299.0, 304.0, 309.0, 307.0, 314.0],
                'highs': [301.0, 306.0, 311.0, 309.0, 316.0],
                'lows': [299.0, 304.0, 309.0, 307.0, 314.0],
                'volumes': [800000] * 5
            },
            'GOOGL': {
                'dates': dates,
                'prices': [120.0, 122.0, 121.0, 123.0, 125.0],
                'opens': [119.0, 121.0, 120.0, 122.0, 124.0],
                'highs': [121.0, 123.0, 122.0, 124.0, 126.0],
                'lows': [119.0, 121.0, 120.0, 122.0, 124.0],
                'volumes': [600000] * 5
            }
        }
        
        # Initial portfolio value (Day 1: AAPL@150 + MSFT@300 + GOOGL@120)
        initial_value = (100.0 * 150.0) + (50.0 * 300.0) + (25.0 * 120.0)  # 33000
        
        snapshots = self.calculator.calculate_historical_performance(
            shares=shares,
            price_data=price_data,
            initial_value=initial_value,
            start_date=dates[0]
        )
        
        # Verify we have snapshots for all dates
        self.assertEqual(len(snapshots), 5)
        
        # Verify values at each date
        expected_values = [
            33000.0,  # Day 0: (100*150) + (50*300) + (25*120) = 33000
            33800.0,  # Day 1: (100*155) + (50*305) + (25*122) = 33800
            33725.0,  # Day 2: (100*152) + (50*310) + (25*121) = 33725
            34275.0,  # Day 3: (100*158) + (50*308) + (25*123) = 34275
            35375.0,  # Day 4: (100*165) + (50*315) + (25*125) = 35375
        ]
        
        for i, snapshot in enumerate(snapshots):
            self.assertAlmostEqual(snapshot.total_value, expected_values[i], places=0,
                                 msg=f"Portfolio value mismatch at date {i}")
            self.assertEqual(snapshot.date, dates[i])
        
        # Verify portfolio value increases overall
        self.assertGreater(snapshots[-1].total_value, snapshots[0].total_value,
                         "Portfolio value should increase over time")


if __name__ == '__main__':
    unittest.main()
