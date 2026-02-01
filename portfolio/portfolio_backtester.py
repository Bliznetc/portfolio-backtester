"""
Interactive portfolio backtesting service.

Allows users to:
- Create portfolios with custom ticker weights
- Backtest performance over different time periods (1d, 1w, 1m, 1y)
- See ROI percentage for each period
- Compare different weight distributions
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .data_fetcher import PriceDataFetcher
from .portfolio_calculator import PortfolioCalculator, PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPerformance:
    """Performance metrics for a portfolio over a time period."""
    period: str  # '1d', '1w', '1m', '1y'
    start_date: datetime
    end_date: datetime
    initial_value: float
    final_value: float
    return_pct: float
    return_absolute: float
    time_series: List[PortfolioSnapshot]  # Daily values for charting


class PortfolioBacktester:
    """
    Main service for portfolio backtesting.
    """
    
    def __init__(self, baseline_amount: float = 10000.0):
        """
        Initialize portfolio backtester.
        
        Args:
            baseline_amount: Initial investment amount for calculations
        """
        self.baseline_amount = baseline_amount
        self.data_fetcher = PriceDataFetcher()
        self.calculator = PortfolioCalculator()
    
    def _get_period_dates(self, period: str) -> Tuple[datetime, datetime]:
        """
        Calculate start and end dates for a time period.
        
        Args:
            period: Time period ('1d', '1w', '1m', '1y', '3y', '5y')
            
        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = datetime.now()
        
        if period == '1d':
            # Look back 4 days to catch last trading day (in case today is weekend)
            start_date = end_date - timedelta(days=4)
        elif period == '1w':
            start_date = end_date - timedelta(weeks=1)
        elif period == '1m':
            start_date = end_date - timedelta(days=30)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '3y':
            start_date = end_date - timedelta(days=365 * 3)
        elif period == '5y':
            start_date = end_date - timedelta(days=365 * 5)
        else:
            raise ValueError(f"Unknown period: {period}")
        
        return start_date, end_date
    
    def _get_resolution_for_period(self, period: str) -> str:
        """
        Determine the data resolution based on the time period.
        
        Args:
            period: Time period ('1d', '1w', '1m', '1y', '3y', '5y')
            
        Returns:
            Resolution string ('1m', '15m', '1h', or '1d')
        """
        if period == '1d':
            return '1m'  # 1 minute for 1 day
        elif period == '1w':
            return '15m'  # 15 minutes for 1 week
        else:
            return '1d'  # 1 day for everything else (1m, 1y, 3y, 5y)
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            weights: Dictionary mapping ticker to weight
            
        Returns:
            Normalized weights dictionary
        """
        total = sum(weights.values())
        if total == 0:
            # Equal distribution if all weights are 0
            num_tickers = len(weights)
            return {ticker: 1.0 / num_tickers for ticker in weights}
        
        return {ticker: weight / total for ticker, weight in weights.items()}
    
    def _filter_price_data_to_last_trading_day(self, price_data: Dict) -> Dict:
        """
        Filter price data to only include the last common trading day across all tickers.
        
        This handles cases where different exchanges (NYSE, LSE, etc.) have different trading days.
        Uses the EARLIEST last trading day to ensure all tickers have data for that day.
        
        Args:
            price_data: Dictionary of price data by ticker
            
        Returns:
            Filtered price_data with only last trading day data
        """
        if not price_data:
            return price_data
        
        # Find the earliest last trading day across all tickers
        # This ensures all tickers have data for that day (handles multi-exchange portfolios)
        last_trading_day = None
        for ticker, data in price_data.items():
            if data['dates']:
                ticker_last_day = data['dates'][-1].date()
                if last_trading_day is None:
                    last_trading_day = ticker_last_day
                else:
                    # Use the minimum (earliest) last trading day
                    last_trading_day = min(last_trading_day, ticker_last_day)
        
        if not last_trading_day:
            logger.warning("No valid trading days found in price data")
            return price_data
        
        # Filter all price data to only include the last common trading day
        filtered_price_data = {}
        for ticker, data in price_data.items():
            filtered_dates = []
            filtered_prices = []
            filtered_opens = []
            filtered_highs = []
            filtered_lows = []
            filtered_volumes = []
            
            for i, date in enumerate(data['dates']):
                if date.date() == last_trading_day:
                    filtered_dates.append(date)
                    filtered_prices.append(data['prices'][i])
                    if 'opens' in data and data['opens']:
                        filtered_opens.append(data['opens'][i])
                    if 'highs' in data and data['highs']:
                        filtered_highs.append(data['highs'][i])
                    if 'lows' in data and data['lows']:
                        filtered_lows.append(data['lows'][i])
                    if 'volumes' in data and data['volumes']:
                        filtered_volumes.append(data['volumes'][i])
            
            filtered_price_data[ticker] = {
                'dates': filtered_dates,
                'prices': filtered_prices,
                'opens': filtered_opens,
                'highs': filtered_highs,
                'lows': filtered_lows,
                'volumes': filtered_volumes
            }
        
        logger.info(f"Filtered to last common trading day: {last_trading_day}")
        return filtered_price_data
    
    def _populate_price_data_for_extended_hours(self, price_data: Dict, period: str) -> Dict:
        """
        Populate price data for extended hours (pre-market and after-hours).
        
        For stocks from different exchanges with different market hours (e.g., NYSE UTC-5 vs LSE UTC),
        this function fills in the gaps:
        - Pre-market hours: use previous day's close price
        - After-hours: use current day's close price
        
        This ensures continuous price data throughout the day for multi-exchange portfolios.
        
        Args:
            price_data: Dictionary of price data by ticker with dates, prices, opens, highs, lows, volumes
            period: Time period ('1d' or '1w' - only these periods have intraday data that needs population)
            
        Returns:
            Price data with extended hours populated
        """
        # Only populate for intraday periods
        if period not in ['1d', '1w']:
            return price_data
        
        if not price_data:
            # throw an error
             raise ValueError("No price data provided for extended hours population")

        
        from datetime import datetime, time, timezone
        import pytz
        
        # Market hours (in UTC for consistency): NYSE 9:30-16:00 UTC (14:30 UTC-5 = 09:30 local)
        market_open = 14.5  # 9:30 AM EDT = 14:30 UTC (14 hours 30 minutes)
        market_close = 21.0  # 4:00 PM EDT = 21:00 UTC (21 hours 0 minutes)
        
        populated_price_data = {}
        
        for ticker, data in price_data.items():
            if not data['dates']:
                populated_price_data[ticker] = data
                continue
            
            # Sort by date to ensure proper ordering
            sorted_indices = sorted(range(len(data['dates'])), key=lambda i: data['dates'][i])
            sorted_dates = [data['dates'][i] for i in sorted_indices]
            sorted_prices = [data['prices'][i] for i in sorted_indices]
            sorted_opens = [data['opens'][i] for i in sorted_indices] if data.get('opens') else []
            sorted_highs = [data['highs'][i] for i in sorted_indices] if data.get('highs') else []
            sorted_lows = [data['lows'][i] for i in sorted_indices] if data.get('lows') else []
            sorted_volumes = [data['volumes'][i] for i in sorted_indices] if data.get('volumes') else []
            
            new_dates = []
            new_prices = []
            new_opens = []
            new_highs = []
            new_lows = []
            new_volumes = []
            
            # Get the first date's day for reference
            first_date = sorted_dates[0]
            current_day = first_date.date()
            
            # Get previous close (day before first data point, if available)
            # For now, we'll use the first available price as pre-market price
            previous_close = sorted_prices[0]
            
            # Find the earliest market hour timestamp for this day
            # This helps us determine when to start adding pre-market data
            earliest_market_time = None
            for i, date in enumerate(sorted_dates):
                if date.date() == current_day:
                    earliest_market_time = date.hour + date.minute / 60.0
                    break
            
            # If there's a gap before the first data point, fill from midnight (00:00 UTC)
            if earliest_market_time is not None and earliest_market_time > 0:
                logger.debug(f"{ticker}: Gap detected - earliest market time {earliest_market_time} > 0 (will populate from midnight)")

                # Create pre-market data points from 00:00 up to the first available data point
                # Use hourly points (00:00, 01:00, ..., up to hour before earliest_market_time)
                for hour in range(0, int(earliest_market_time)):
                    pre_market_time = first_date.replace(hour=hour, minute=0, second=0, microsecond=0)

                    # Only add if it doesn't overlap with actual data
                    if pre_market_time < sorted_dates[0]:
                        new_dates.append(pre_market_time)
                        new_prices.append(previous_close)
                        new_opens.append(previous_close)
                        new_highs.append(previous_close)
                        new_lows.append(previous_close)
                        new_volumes.append(0)
            
            # Add all market hours data
            for i, date in enumerate(sorted_dates):
                new_dates.append(date)
                new_prices.append(sorted_prices[i])
                new_opens.append(sorted_opens[i] if sorted_opens else sorted_prices[i])
                new_highs.append(sorted_highs[i] if sorted_highs else sorted_prices[i])
                new_lows.append(sorted_lows[i] if sorted_lows else sorted_prices[i])
                new_volumes.append(sorted_volumes[i] if sorted_volumes else 0)
            
            # After-market: fill until end of day (23:59)
            if sorted_dates:
                last_date = sorted_dates[-1]
                last_price = sorted_prices[-1]
                last_hour = last_date.hour + last_date.minute / 60.0
                
                if last_hour < 23.5:  # If we end before 23:30
                    logger.debug(f"{ticker}: Adding after-hours data from {last_hour} to 23:59")
                    
                    # Add after-hours data points (hourly)
                    for hour in range(int(last_hour) + 1, 24):
                        after_market_time = last_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                        
                        # Only add if after last data point
                        if after_market_time > sorted_dates[-1]:
                            new_dates.append(after_market_time)
                            new_prices.append(last_price)
                            new_opens.append(last_price)
                            new_highs.append(last_price)
                            new_lows.append(last_price)
                            new_volumes.append(0)
            
            populated_price_data[ticker] = {
                'dates': new_dates,
                'prices': new_prices,
                'opens': new_opens,
                'highs': new_highs,
                'lows': new_lows,
                'volumes': new_volumes
            }
            
            logger.debug(f"{ticker}: Extended hours population - {len(sorted_dates)} original → {len(new_dates)} populated data points")
        
        return populated_price_data
    
    def _normalize_price_data_timezones(self, price_data: Dict) -> Dict:
        """
        Normalize all timestamps in price_data to UTC (naive datetime).
        
        This ensures consistent timezone handling across all tickers and prevents
        issues with multi-exchange portfolios where data comes in different timezones.
        
        Args:
            price_data: Dictionary of price data by ticker with dates
            
        Returns:
            Price data with all timestamps normalized to UTC (timezone-naive)
        """
        if not price_data:
            return price_data
        
        from datetime import timezone as tz
        
        normalized_price_data = {}
        
        for ticker, data in price_data.items():
            normalized_dates = []
            
            for date in data['dates']:
                # Convert to UTC and make naive
                if date.tzinfo is not None:
                    utc_date = date.astimezone(tz.utc).replace(tzinfo=None)
                else:
                    utc_date = date
                normalized_dates.append(utc_date)
            
            normalized_price_data[ticker] = {
                'dates': normalized_dates,
                'prices': data['prices'],
                'opens': data['opens'],
                'highs': data['highs'],
                'lows': data['lows'],
                'volumes': data['volumes']
            }
            
            logger.debug(f"{ticker}: Normalized {len(normalized_dates)} timestamps to UTC")
        
        return normalized_price_data
    
    def backtest_portfolio(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        periods: List[str] = ['1d', '1w', '1m', '1y', '3y', '5y']
    ) -> Dict[str, PortfolioPerformance]:
        """
        Backtest portfolio performance for multiple time periods.
        
        Args:
            tickers: List of stock symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            weights: Dictionary mapping ticker to target weight (e.g., {'AAPL': 0.2, 'MSFT': 0.3, 'GOOGL': 0.5})
            periods: List of time periods to analyze (e.g., ['1d', '1w', '1m', '1y', '3y', '5y'])
            
        Returns:
            Dictionary mapping period to PortfolioPerformance
        """
        # Normalize weights
        weights = self._normalize_weights(weights)
        
        # Ensure all tickers have weights (default to 0 if missing)
        for ticker in tickers:
            if ticker not in weights:
                weights[ticker] = 0.0
        
        results = {}
        
        for period in periods:
            try:
                start_date, end_date = self._get_period_dates(period)
                
                logger.info(f"Backtesting {period} period: {start_date.date()} to {end_date.date()}")
                
                # Determine resolution based on period
                resolution = self._get_resolution_for_period(period)
                
                # Fetch historical prices for all tickers concurrently
                price_data = self.data_fetcher.fetch_historical_prices(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=resolution
                )
                
                if not price_data:
                    logger.warning(f"No price data available for period {period}")
                    continue
                
                # Normalize all timestamps to UTC FIRST (before any filtering)
                # This ensures we're working with consistent timezones
                price_data = self._normalize_price_data_timezones(price_data)
                
                # For 1D period, filter to only the last common trading day across all tickers
                # (filter AFTER normalization to avoid timezone-related day shifts)
                if period == '1d' and price_data:
                    price_data = self._filter_price_data_to_last_trading_day(price_data)
                
                # For 1D and 1W periods, populate extended hours (pre-market and after-hours)
                # This handles multi-exchange portfolios with different market hours
                if period in ['1d', '1w'] and price_data:
                    price_data = self._populate_price_data_for_extended_hours(price_data, period)
                
                
                
                # Get start prices for each ticker
                start_prices = {}
                for ticker in tickers:
                    if ticker in price_data and price_data[ticker]['dates']:
                        # Get first available price (now first of filtered day)
                        start_prices[ticker] = price_data[ticker]['prices'][0]
                    else:
                        # Try to fetch price on start date
                        price = self.data_fetcher.get_price_on_date(ticker, start_date)
                        start_prices[ticker] = price
                
                # Calculate initial shares
                shares = self.calculator.calculate_initial_shares(
                    tickers=tickers,
                    weights=weights,
                    initial_value=self.baseline_amount,
                    start_prices=start_prices
                )
                
                # Calculate historical performance
                snapshots = self.calculator.calculate_historical_performance(
                    shares=shares,
                    price_data=price_data,
                    initial_value=self.baseline_amount,
                    start_date=start_date
                )
                
                # Debug logging for each period
                if snapshots:
                    unique_dates = len(set(s.date for s in snapshots))
                    logger.info(f"=== {period.upper()} Period Debug ===")
                    logger.info(f"Total snapshots: {len(snapshots)}")
                    logger.info(f"Unique timestamps: {unique_dates}")
                    logger.info(f"First snapshot: {snapshots[0].date} | Value: ${snapshots[0].total_value:.2f}")
                    logger.info(f"Last snapshot: {snapshots[-1].date} | Value: ${snapshots[-1].total_value:.2f}")
                    
                    # Log first 5 and last 5 snapshots
                    logger.info(f"First 5 snapshots:")
                    for i, snap in enumerate(snapshots[:5]):
                        logger.info(f"  {i}: {snap.date} | ${snap.total_value:.2f}")
                    
                    if len(snapshots) > 10:
                        logger.info(f"Last 5 snapshots:")
                        for i, snap in enumerate(snapshots[-5:], start=len(snapshots)-5):
                            logger.info(f"  {i}: {snap.date} | ${snap.total_value:.2f}")
                    
                    # Check for duplicate timestamps
                    timestamps = [s.date for s in snapshots]
                    duplicates = [ts for ts in set(timestamps) if timestamps.count(ts) > 1]
                    if duplicates:
                        logger.warning(f"Found {len(duplicates)} duplicate timestamps!")
                        for ts in duplicates[:5]:  # Show first 5 duplicates
                            count = timestamps.count(ts)
                            logger.warning(f"  {ts} appears {count} times")
                
                if not snapshots:
                    logger.warning(f"No snapshots calculated for period {period}")
                    continue
                
                # Get final values
                final_snapshot = snapshots[-1]
                final_value = final_snapshot.total_value
                return_pct = final_snapshot.return_pct
                return_absolute = final_value - self.baseline_amount
                
                performance = PortfolioPerformance(
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    initial_value=self.baseline_amount,
                    final_value=final_value,
                    return_pct=return_pct,
                    return_absolute=return_absolute,
                    time_series=snapshots
                )
                
                results[period] = performance
                logger.info(f"✓ {period}: {return_pct:.2f}% return (${self.baseline_amount:.2f} → ${final_value:.2f})")
                
            except Exception as e:
                logger.error(f"Error backtesting period {period}: {e}")
                continue
        
        return results
    
    def compare_distributions(
        self,
        tickers: List[str],
        distributions: List[Dict[str, float]],
        periods: List[str] = ['1d', '1w', '1m', '1y']
    ) -> Dict[str, List[PortfolioPerformance]]:
        """
        Compare multiple portfolio distributions.
        
        Args:
            tickers: List of stock symbols
            distributions: List of weight dictionaries to compare
            periods: List of time periods to analyze
            
        Returns:
            Dictionary mapping period to list of PortfolioPerformance for each distribution
        """
        comparison = {}
        
        for period in periods:
            comparison[period] = []
            
            for i, weights in enumerate(distributions):
                try:
                    results = self.backtest_portfolio(tickers, weights, [period])
                    if period in results:
                        comparison[period].append(results[period])
                except Exception as e:
                    logger.error(f"Error comparing distribution {i} for period {period}: {e}")
        
        return comparison
    
    def get_equal_weights(self, tickers: List[str]) -> Dict[str, float]:
        """
        Generate equal weights for a list of tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary with equal weights
        """
        weight = 1.0 / len(tickers) if tickers else 0.0
        return {ticker: weight for ticker in tickers}
