#!/usr/bin/env python3
"""
Test script for portfolio backtester.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import unittest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio.portfolio_backtester import PortfolioBacktester
from config.settings import FINNHUB_API_KEY


# Unit tests for extended hours population
class TestExtendedHoursPopulation(unittest.TestCase):
    """Test suite for extended hours price data population."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backtester = PortfolioBacktester(baseline_amount=10000.0)
    
    def test_populate_price_data_1d_with_gap_before_open(self):
        """Test population fills pre-market gap when first data point is after market open."""
        # Create mock price data for 1D with market hours only (14:30-21:00 UTC ~ 9:30-16:00 EDT)
        start_time = datetime(2026, 1, 23, 16, 30)  # 11:30 EDT - 2.5 hours after open
        
        price_data = {
            'AAPL': {
                'dates': [
                    start_time,
                    start_time + timedelta(hours=1),
                    start_time + timedelta(hours=2),
                ],
                'prices': [150.0, 151.0, 152.0],
                'opens': [150.0, 151.0, 152.0],
                'highs': [150.5, 151.5, 152.5],
                'lows': [149.5, 150.5, 151.5],
                'volumes': [1000000, 900000, 800000]
            }
        }
        
        # Populate for 1D
        result = self.backtester._populate_price_data_for_extended_hours(price_data, '1d')
        
        # Verify result has more data points (added pre-market)
        self.assertGreater(len(result['AAPL']['dates']), len(price_data['AAPL']['dates']))
        
        # Verify early data points have pre-market price (150.0 - first price)
        pre_market_prices = [result['AAPL']['prices'][i] for i in range(len(result['AAPL']['dates']))
                             if result['AAPL']['dates'][i] < start_time]
        self.assertTrue(all(p == 150.0 for p in pre_market_prices))
        
        # Verify volumes are 0 for pre-market
        pre_market_volumes = [result['AAPL']['volumes'][i] for i in range(len(result['AAPL']['dates']))
                              if result['AAPL']['dates'][i] < start_time]
        self.assertTrue(all(v == 0 for v in pre_market_volumes))

        # Verify coverage from midnight to end of day (00:00 to at least 23:00)
        hours = [d.hour for d in result['AAPL']['dates']]
        self.assertEqual(min(hours), 0)
        self.assertGreaterEqual(max(hours), 23)
        
        print("✓ test_populate_price_data_1d_with_gap_before_open passed")
    
    def test_populate_price_data_1d_with_gap_after_close(self):
        """Test population fills after-hours gap when last data point is before market close."""
        # Create mock price data ending before market close (21:00 UTC)
        start_time = datetime(2026, 1, 23, 14, 30)  # 9:30 EDT
        end_time = datetime(2026, 1, 23, 19, 0)      # 14:00 EDT - 2 hours before close
        
        price_data = {
            'AAPL': {
                'dates': [
                    start_time,
                    start_time + timedelta(hours=2),
                    end_time,
                ],
                'prices': [150.0, 151.0, 152.0],
                'opens': [150.0, 151.0, 152.0],
                'highs': [150.5, 151.5, 152.5],
                'lows': [149.5, 150.5, 151.5],
                'volumes': [1000000, 900000, 800000]
            }
        }
        
        # Populate for 1D
        result = self.backtester._populate_price_data_for_extended_hours(price_data, '1d')
        
        # Verify result has more data points (added after-hours)
        self.assertGreater(len(result['AAPL']['dates']), len(price_data['AAPL']['dates']))
        
        # Verify after-market data points have close price (152.0 - last price)
        after_market_prices = [result['AAPL']['prices'][i] for i in range(len(result['AAPL']['dates']))
                               if result['AAPL']['dates'][i] > end_time]
        self.assertTrue(all(p == 152.0 for p in after_market_prices))
        
        # Verify volumes are 0 for after-market
        after_market_volumes = [result['AAPL']['volumes'][i] for i in range(len(result['AAPL']['dates']))
                                if result['AAPL']['dates'][i] > end_time]
        self.assertTrue(all(v == 0 for v in after_market_volumes))
        
        print("✓ test_populate_price_data_1d_with_gap_after_close passed")

        # Verify coverage from midnight to end of day (00:00 to at least 23:00)
        hours = [d.hour for d in result['AAPL']['dates']]
        self.assertEqual(min(hours), 0)
        self.assertGreaterEqual(max(hours), 23)
    
    def test_populate_price_data_1w_multi_exchange(self):
        """Test population handles multiple tickers from different exchanges."""
        # NYSE ticker - market hours 14:30-21:00 UTC (9:30-16:00 EDT)
        nyse_start = datetime(2026, 1, 23, 16, 30)  # 11:30 EDT
        
        # LSE ticker - market hours 8:00-16:30 UTC (8:00-16:30 GMT)
        lse_start = datetime(2026, 1, 23, 8, 0)     # 8:00 GMT
        
        price_data = {
            'AAPL': {  # NYSE
                'dates': [nyse_start, nyse_start + timedelta(hours=1)],
                'prices': [150.0, 151.0],
                'opens': [150.0, 151.0],
                'highs': [150.5, 151.5],
                'lows': [149.5, 150.5],
                'volumes': [1000000, 900000]
            },
            'ASML': {  # LSE (Euronext actually, but similar hours)
                'dates': [lse_start, lse_start + timedelta(hours=1)],
                'prices': [600.0, 601.0],
                'opens': [600.0, 601.0],
                'highs': [600.5, 601.5],
                'lows': [599.5, 600.5],
                'volumes': [500000, 450000]
            }
        }
        
        # Populate for 1W
        result = self.backtester._populate_price_data_for_extended_hours(price_data, '1w')
        
        # Verify both tickers have populated data
        self.assertGreater(len(result['AAPL']['dates']), len(price_data['AAPL']['dates']))
        self.assertGreater(len(result['ASML']['dates']), len(price_data['ASML']['dates']))
        
        # Verify AAPL has pre-market data (after 14:30 but before first point at 16:30)
        aapl_early = [result['AAPL']['dates'][i] for i in range(len(result['AAPL']['dates']))
                      if result['AAPL']['dates'][i] < nyse_start]
        self.assertGreater(len(aapl_early), 0)

        # Verify coverage from midnight to end of day (00:00 to at least 23:00) for both tickers
        aapl_hours = [d.hour for d in result['AAPL']['dates']]
        asml_hours = [d.hour for d in result['ASML']['dates']]
        self.assertEqual(min(aapl_hours), 0)
        self.assertGreaterEqual(max(aapl_hours), 23)
        self.assertEqual(min(asml_hours), 0)
        self.assertGreaterEqual(max(asml_hours), 23)

        # Verify ASML price at 15:00 UTC + AAPL price at 15:00 UTC
        aapl_price_at_15 = None
        asml_price_at_15 = None
        for i, d in enumerate(result['AAPL']['dates']):
            if d.hour == 15 and d.date() == datetime(2026, 1, 23).date():
                aapl_price_at_15 = result['AAPL']['prices'][i]
                break
        for i, d in enumerate(result['ASML']['dates']):
            if d.hour == 15 and d.date() == datetime(2026, 1, 23).date():
                asml_price_at_15 = result['ASML']['prices'][i]
                break 
        self.assertEqual(aapl_price_at_15 + asml_price_at_15, 751.0)  # 150.0 + 601.0

        print("✓ test_populate_price_data_1w_multi_exchange passed")
    
    def test_populate_price_data_no_data(self):
        """Test population raises error for empty price data."""
        price_data = {}
        
        with self.assertRaises((ValueError, KeyError)) as context:
            self.backtester._populate_price_data_for_extended_hours(price_data, '1d')
        
        print("✓ test_populate_price_data_no_data passed")
    
    def test_populate_price_data_wrong_period(self):
        """Test population skips non-intraday periods."""
        price_data = {
            'AAPL': {
                'dates': [datetime(2026, 1, 23, 14, 30)],
                'prices': [150.0],
                'opens': [150.0],
                'highs': [150.5],
                'lows': [149.5],
                'volumes': [1000000]
            }
        }
        
        # Populate for daily (should not modify)
        result = self.backtester._populate_price_data_for_extended_hours(price_data, '1m')
        
        # Should return same data unchanged
        self.assertEqual(len(result['AAPL']['dates']), len(price_data['AAPL']['dates']))
        
        print("✓ test_populate_price_data_wrong_period passed")
    
    def test_populate_price_data_preserves_market_hours(self):
        """Test that population preserves original market hours data."""
        start_time = datetime(2026, 1, 23, 14, 30)  # 9:30 EDT
        
        original_prices = [150.0, 151.0, 152.0]
        price_data = {
            'AAPL': {
                'dates': [start_time + timedelta(hours=i) for i in range(3)],
                'prices': original_prices,
                'opens': [150.0, 151.0, 152.0],
                'highs': [150.5, 151.5, 152.5],
                'lows': [149.5, 150.5, 151.5],
                'volumes': [1000000, 900000, 800000]
            }
        }
        
        result = self.backtester._populate_price_data_for_extended_hours(price_data, '1d')
        
        # Find indices of original timestamps in result
        original_indices = []
        for orig_date in price_data['AAPL']['dates']:
            for i, res_date in enumerate(result['AAPL']['dates']):
                if res_date == orig_date:
                    original_indices.append(i)
                    break
        
        # Verify original prices are preserved
        result_original_prices = [result['AAPL']['prices'][i] for i in original_indices]
        self.assertEqual(result_original_prices, original_prices)
        
        print("✓ test_populate_price_data_preserves_market_hours passed")


# Unit tests for timezone normalization and performance
class TestTimezoneNormalizationAndPerformance(unittest.TestCase):
    """Test suite for timezone normalization and performance calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backtester = PortfolioBacktester(baseline_amount=10000.0)
    
    def test_normalize_timezones_utc_to_naive(self):
        """Test that UTC-aware timestamps are converted to naive UTC."""
        from datetime import timezone as tz
        
        # Create price data with UTC-aware timestamps
        utc_time = datetime(2026, 1, 26, 14, 30, tzinfo=tz.utc)
        
        price_data = {
            'AAPL': {
                'dates': [utc_time, utc_time + timedelta(hours=1)],
                'prices': [150.0, 151.0],
                'opens': [150.0, 151.0],
                'highs': [150.5, 151.5],
                'lows': [149.5, 150.5],
                'volumes': [1000000, 900000]
            }
        }
        
        result = self.backtester._normalize_price_data_timezones(price_data)
        
        # Verify timestamps are naive (no timezone info)
        for date in result['AAPL']['dates']:
            self.assertIsNone(date.tzinfo)
        
        # Verify timestamps are still in UTC (same hour)
        self.assertEqual(result['AAPL']['dates'][0].hour, 14)
        self.assertEqual(result['AAPL']['dates'][0].minute, 30)
        
        print("✓ test_normalize_timezones_utc_to_naive passed")
    
    def test_normalize_timezones_est_to_utc(self):
        """Test that EST-aware timestamps are converted to naive UTC correctly."""
        import pytz
        
        # Create price data with EST timestamps
        est = pytz.timezone('US/Eastern')
        est_time = est.localize(datetime(2026, 1, 26, 9, 30))  # 9:30 AM EST
        
        price_data = {
            'AAPL': {
                'dates': [est_time, est_time + timedelta(hours=1)],
                'prices': [150.0, 151.0],
                'opens': [150.0, 151.0],
                'highs': [150.5, 151.5],
                'lows': [149.5, 150.5],
                'volumes': [1000000, 900000]
            }
        }
        
        result = self.backtester._normalize_price_data_timezones(price_data)
        
        # Verify timestamps are naive
        for date in result['AAPL']['dates']:
            self.assertIsNone(date.tzinfo)
        
        # Verify conversion to UTC: 9:30 EST = 14:30 UTC (EST is UTC-5)
        self.assertEqual(result['AAPL']['dates'][0].hour, 14)
        self.assertEqual(result['AAPL']['dates'][0].minute, 30)
        
        print("✓ test_normalize_timezones_est_to_utc passed")
    
    def test_normalize_timezones_mixed_timezone_data(self):
        """Test normalization with mixed timezones (NYSE EST and LSE UTC)."""
        import pytz
        
        est = pytz.timezone('US/Eastern')
        utc = pytz.timezone('UTC')
        
        # NYSE data in EST
        nyse_time = est.localize(datetime(2026, 1, 26, 9, 30))
        # LSE data in UTC
        lse_time = utc.localize(datetime(2026, 1, 26, 8, 0))
        
        price_data = {
            'AAPL': {  # NYSE - EST
                'dates': [nyse_time, nyse_time + timedelta(hours=1)],
                'prices': [150.0, 151.0],
                'opens': [150.0, 151.0],
                'highs': [150.5, 151.5],
                'lows': [149.5, 150.5],
                'volumes': [1000000, 900000]
            },
            'ASML': {  # LSE - UTC
                'dates': [lse_time, lse_time + timedelta(hours=1)],
                'prices': [600.0, 601.0],
                'opens': [600.0, 601.0],
                'highs': [600.5, 601.5],
                'lows': [599.5, 600.5],
                'volumes': [500000, 450000]
            }
        }
        
        result = self.backtester._normalize_price_data_timezones(price_data)
        
        # Both should be naive UTC
        self.assertIsNone(result['AAPL']['dates'][0].tzinfo)
        self.assertIsNone(result['ASML']['dates'][0].tzinfo)
        
        # AAPL: 9:30 EST = 14:30 UTC
        self.assertEqual(result['AAPL']['dates'][0].hour, 14)
        # ASML: 8:00 UTC = 8:00 UTC
        self.assertEqual(result['ASML']['dates'][0].hour, 8)
        
        # Both on same date
        self.assertEqual(result['AAPL']['dates'][0].date(), datetime(2026, 1, 26).date())
        self.assertEqual(result['ASML']['dates'][0].date(), datetime(2026, 1, 26).date())
        
        print("✓ test_normalize_timezones_mixed_timezone_data passed")
    
    def test_no_day_bleed_after_normalization(self):
        """Test that after-hours timestamps don't bleed into next day after normalization."""
        import pytz
        
        # Create EST data that ends at 8 PM EST (which is 1 AM UTC next day)
        est = pytz.timezone('US/Eastern')
        est_time_8pm = est.localize(datetime(2026, 1, 26, 20, 0))
        
        price_data = {
            'AAPL': {
                'dates': [est_time_8pm],
                'prices': [150.0],
                'opens': [150.0],
                'highs': [150.5],
                'lows': [149.5],
                'volumes': [1000000]
            }
        }
        
        # Normalize first
        normalized = self.backtester._normalize_price_data_timezones(price_data)
        
        # Populate extended hours (this should add hours 1-23 on Jan 27 UTC)
        result = self.backtester._populate_price_data_for_extended_hours(normalized, '1d')
        
        # Check all dates are on Jan 27 (since 8 PM EST Jan 26 = 1 AM UTC Jan 27)
        dates = result['AAPL']['dates']
        unique_days = set(d.date() for d in dates)
        
        # Should all be on Jan 27 UTC
        self.assertEqual(len(unique_days), 1)
        self.assertEqual(list(unique_days)[0], datetime(2026, 1, 27).date())
        
        # Max hour should be 23
        max_hour = max(d.hour for d in dates)
        self.assertLessEqual(max_hour, 23)
        
        print("✓ test_no_day_bleed_after_normalization passed")
    
    def test_filter_selects_min_last_trading_day(self):
        """Test that filter selects the earliest last trading day across all tickers."""
        # Ticker 1 has data through Jan 27
        # Ticker 2 has data only through Jan 26
        # Should select Jan 26 as the last common trading day
        
        price_data = {
            'AAPL': {
                'dates': [
                    datetime(2026, 1, 26, 14, 0),
                    datetime(2026, 1, 26, 15, 0),
                    datetime(2026, 1, 27, 14, 0),  # Has Jan 27 data
                ],
                'prices': [150.0, 151.0, 152.0],
                'opens': [150.0, 151.0, 152.0],
                'highs': [150.5, 151.5, 152.5],
                'lows': [149.5, 150.5, 151.5],
                'volumes': [1000000, 900000, 800000]
            },
            'MSFT': {
                'dates': [
                    datetime(2026, 1, 26, 14, 0),
                    datetime(2026, 1, 26, 15, 0),  # Only has Jan 26 data
                ],
                'prices': [300.0, 301.0],
                'opens': [300.0, 301.0],
                'highs': [300.5, 301.5],
                'lows': [299.5, 300.5],
                'volumes': [2000000, 1900000]
            }
        }
        
        result = self.backtester._filter_price_data_to_last_trading_day(price_data)
        
        # Should only have Jan 26 data for both tickers
        for ticker in ['AAPL', 'MSFT']:
            dates = [d.date() for d in result[ticker]['dates']]
            unique_days = set(dates)
            self.assertEqual(len(unique_days), 1)
            self.assertEqual(list(unique_days)[0], datetime(2026, 1, 26).date())
        
        # AAPL should have 2 data points (Jan 27 filtered out)
        self.assertEqual(len(result['AAPL']['dates']), 2)
        # MSFT should have 2 data points
        self.assertEqual(len(result['MSFT']['dates']), 2)
        
        print("✓ test_filter_selects_min_last_trading_day passed")
    
    def test_performance_calculation_preserves_prices(self):
        """Test that performance calculation correctly uses normalized prices."""
        # Create simple price data with known values
        price_data = {
            'AAPL': {
                'dates': [
                    datetime(2026, 1, 26, 14, 0),
                    datetime(2026, 1, 26, 15, 0),
                    datetime(2026, 1, 26, 16, 0),
                ],
                'prices': [100.0, 110.0, 120.0],  # 10% then 9.09% gains
                'opens': [100.0, 110.0, 120.0],
                'highs': [100.0, 110.0, 120.0],
                'lows': [100.0, 110.0, 120.0],
                'volumes': [1000000, 1000000, 1000000]
            }
        }
        
        # Calculate with 100% allocation to AAPL
        shares = self.backtester.calculator.calculate_initial_shares(
            tickers=['AAPL'],
            weights={'AAPL': 1.0},
            initial_value=10000.0,
            start_prices={'AAPL': 100.0}
        )
        
        # Should have 100 shares (10000 / 100)
        self.assertEqual(shares['AAPL'], 100.0)
        
        # Calculate performance
        snapshots = self.backtester.calculator.calculate_historical_performance(
            shares=shares,
            price_data=price_data,
            initial_value=10000.0,
            start_date=datetime(2026, 1, 26)
        )
        
        # Should have 3 snapshots
        self.assertEqual(len(snapshots), 3)
        
        # Check values
        self.assertAlmostEqual(snapshots[0].total_value, 10000.0, places=2)  # 100 * 100
        self.assertAlmostEqual(snapshots[1].total_value, 11000.0, places=2)  # 100 * 110
        self.assertAlmostEqual(snapshots[2].total_value, 12000.0, places=2)  # 100 * 120
        
        # Check returns
        self.assertAlmostEqual(snapshots[0].return_pct, 0.0, places=2)
        self.assertAlmostEqual(snapshots[1].return_pct, 10.0, places=2)
        self.assertAlmostEqual(snapshots[2].return_pct, 20.0, places=2)
        
        print("✓ test_performance_calculation_preserves_prices passed")


if __name__ == '__main__':
    # Run unit tests first
    print("=" * 80)
    print("EXTENDED HOURS POPULATION UNIT TESTS")
    print("=" * 80)
    
    # Create test suites
    loader = unittest.TestLoader()
    suite1 = loader.loadTestsFromTestCase(TestExtendedHoursPopulation)
    suite2 = loader.loadTestsFromTestCase(TestTimezoneNormalizationAndPerformance)
    
    # Combine suites
    combined_suite = unittest.TestSuite([suite1, suite2])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Exit with status
    sys.exit(0 if result.wasSuccessful() else 1)
