"""
Historical price data fetcher with concurrent fetching support.

Uses yfinance for historical price data (free, no API key required).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf
except ImportError:
    yf = None
    logging.warning("yfinance not installed. Install with: pip install yfinance")

try:
    import pytz
except ImportError:
    pytz = None

logger = logging.getLogger(__name__)


class PriceDataFetcher:
    """
    Fetches historical price data for multiple tickers concurrently.
    
    Uses yfinance library which provides free historical stock data.
    """
    
    def __init__(self, timezone: Optional[str] = None):
        """
        Initialize price data fetcher.
        
        Args:
            timezone: Optional timezone string (e.g., 'UTC', 'America/New_York', 'Europe/London')
                     If None, timestamps are returned in their original timezone from yfinance.
                     Requires pytz library if specified.
        """
        if yf is None:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        self.timezone = None
        if timezone:
            if pytz is None:
                raise ImportError("pytz is required for timezone conversion. Install with: pip install pytz")
            try:
                self.timezone = pytz.timezone(timezone)
                logger.info(f"Using timezone: {timezone}")
            except Exception as e:
                logger.warning(f"Invalid timezone '{timezone}': {e}. Using original timezone from data.")
                self.timezone = None
    
    def validate_ticker(self, symbol: str, days_back: int = 30) -> bool:
        """
        Validate if a ticker exists and has recent price data.
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days back to check for data
            
        Returns:
            True if ticker is valid and has data, False otherwise
        """
        if yf is None:
            logger.error("yfinance not installed, cannot validate ticker")
            return False
        
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Try to fetch historical data
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No historical data found for ticker {symbol}")
                return False
            
            # Check if we have at least some valid price data
            if hist['Close'].isna().all():
                logger.warning(f"No valid price data for ticker {symbol}")
                return False
                
            logger.info(f"Ticker {symbol} validated successfully")
            return True
        except Exception as e:
            logger.warning(f"Error validating ticker {symbol}: {e}")
            return False
    
    def _fetch_single_ticker(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        resolution: str = '1d'
    ) -> Optional[Dict[str, List]]:
        """
        Fetch historical data for a single ticker using yfinance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for historical data
            end_date: End date for historical data
            resolution: Data resolution. Must be one of: '1m', '15m', '1h', '1d', '1w', '1mo'
            
        Returns:
            Dictionary with 'dates' (datetime objects), 'prices' (close prices), 
            'opens', 'highs', 'lows', 'volumes'
            or None if error
        """
        try:
            # Map resolution to yfinance interval format
            resolution_map = {
                '1m': '1m',
                '15m': '15m',
                '1h': '1h',
                '1d': '1d',
                '1w': '1wk',
                '1mo': '1mo'
            }
            
            yf_interval = resolution_map.get(resolution, '1d')
            if resolution not in resolution_map:
                logger.warning(f"Unknown resolution '{resolution}', defaulting to '1d'")
            
            ticker = yf.Ticker(symbol)
            
            # Fetch data based on resolution type
            if yf_interval in ['1m', '1h']:
                hist = self._fetch_intraday_data(ticker, yf_interval, start_date, end_date)
            else:
                hist = ticker.history(start=start_date, end=end_date, interval=yf_interval)
            
            if hist.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Extract and convert data
            return self._extract_price_data(hist, symbol)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_historical_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str = '1d',
        max_workers: int = 5
    ) -> Dict[str, Dict[str, List]]:
        """
        Fetch historical prices for multiple tickers concurrently.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            resolution: Data resolution. Must be one of: '1m', '1h', '1d', '1w', '1mo'
            max_workers: Maximum number of concurrent requests
            
        Returns:
            Dictionary mapping ticker to price data:
            {
                'AAPL': {
                    'dates': [datetime, ...],
                    'prices': [float, ...],
                    ...
                },
                ...
            }
        """
        logger.info(f"Fetching historical prices for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}")
        
        results = {}
        
        # Fetch data concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    self._fetch_single_ticker,
                    ticker,
                    start_date,
                    end_date,
                    resolution
                ): ticker
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        results[ticker] = data
                        logger.info(f"✓ Fetched {len(data['prices'])} data points for {ticker}")
                    else:
                        logger.warning(f"✗ No data for {ticker}")
                except Exception as e:
                    logger.error(f"✗ Error fetching {ticker}: {e}")
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(tickers)} tickers")
        return results
    
    def get_price_on_date(
        self,
        ticker: str,
        target_date: datetime,
        lookback_days: int = 5
    ) -> Optional[float]:
        """
        Get the price of a ticker on a specific date.
        If exact date not available, looks back up to lookback_days.
        
        Args:
            ticker: Stock symbol
            target_date: Date to get price for
            lookback_days: Maximum days to look back if exact date not available
            
        Returns:
            Price on that date or None if not found
        """
        start_date = target_date - timedelta(days=lookback_days)
        end_date = target_date + timedelta(days=1)  # Include target date
        
        data = self._fetch_single_ticker(ticker, start_date, end_date, '1d')
        
        if not data or not data['dates']:
            return None
        
        # Find closest date to target_date
        dates = data['dates']
        prices = data['prices']
        
        # Find exact match or closest before target_date
        for i, date in enumerate(dates):
            if date.date() == target_date.date():
                return prices[i]
            elif date.date() > target_date.date() and i > 0:
                # Return previous day's price
                return prices[i - 1]
        
        # If target_date is after all dates, return last price
        if dates and dates[-1].date() < target_date.date():
            return prices[-1]
        
        return None
    
    def _fetch_intraday_data(
        self,
        ticker: yf.Ticker,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Fetch intraday data (1m, 15m, or 1h) with proper period handling.
        
        Args:
            ticker: yfinance Ticker object
            interval: '1m', '15m', or '1h'
            start_date: Requested start date
            end_date: Requested end date
            
        Returns:
            DataFrame with historical data
        """
        days_diff = (end_date - start_date).days
        
        # Apply yfinance limitations
        if interval == '1m' and days_diff > 7:
            logger.warning(f"1m resolution limited to 7 days. Requested {days_diff} days, using last 7 days")
            start_date = end_date - timedelta(days=7)
            days_diff = 7
        elif interval in ['15m', '5m'] and days_diff > 60:
            logger.warning(f"{interval} resolution limited to 60 days. Requested {days_diff} days, using last 60 days")
            start_date = end_date - timedelta(days=60)
            days_diff = 60
        elif interval == '1h' and days_diff > 730:
            logger.warning(f"1h resolution limited to 730 days. Requested {days_diff} days, using last 730 days")
            start_date = end_date - timedelta(days=730)
            days_diff = 730
        
        # Determine period parameter
        period = self._get_period_for_intraday(interval, days_diff)
        
        # Fetch data
        hist = ticker.history(period=period, interval=interval)
        
        if not hist.empty:
            hist = self._filter_intraday_by_date_range(hist, start_date, end_date)
        
        return hist
    
    def _get_period_for_intraday(self, interval: str, days_diff: int) -> str:
        """Get yfinance period parameter for intraday data."""
        if interval == '1m':
            if days_diff <= 1:
                return '1d'
            elif days_diff <= 5:
                return '5d'
            else:
                return '7d'
        elif interval == '15m':
            if days_diff <= 1:
                return '1d'
            elif days_diff <= 5:
                return '5d'
            elif days_diff <= 30:
                return '1mo'
            else:
                return '2mo'
        else:  # 1h
            if days_diff <= 1:
                return '1d'
            elif days_diff <= 5:
                return '5d'
            elif days_diff <= 30:
                return '1mo'
            elif days_diff <= 90:
                return '3mo'
            else:
                return '1y'
    
    def _filter_intraday_by_date_range(self, hist, start_date: datetime, end_date: datetime):
        """
        Filter intraday data to requested date range.
        Returns all data if filtering would remove too much (>90%).
        """
        try:
            # Normalize timezones for comparison
            start_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
            end_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
            
            # Convert hist index to timezone-naive
            hist_index_naive = self._normalize_datetime_index(hist.index)
            
            # Filter by time range
            mask = (hist_index_naive >= start_naive) & (hist_index_naive <= end_naive)
            filtered_hist = hist[mask]
            
            # If filtering removed too much, check if data is from requested date
            if len(filtered_hist) < len(hist) * 0.1 and len(hist) > 10:
                hist_date = hist_index_naive[0].date()
                start_date_only = start_naive.date()
                end_date_only = end_naive.date()
                
                # If data is from requested date range, return all of it
                if start_date_only <= hist_date <= end_date_only:
                    logger.debug(f"Filtering removed {len(hist) - len(filtered_hist)}/{len(hist)} points, "
                               f"but data is from requested date ({hist_date}), returning all {len(hist)} points")
                    return hist
                else:
                    return filtered_hist
            
            return filtered_hist
            
        except Exception as e:
            logger.debug(f"Could not filter intraday data: {e}, returning all {len(hist)} available data points")
            return hist
    
    def _normalize_datetime_index(self, index):
        """Convert timezone-aware DatetimeIndex to timezone-naive."""
        import pandas as pd
        
        if hasattr(index, 'tz') and index.tz is not None:
            return index.tz_localize(None)
        elif len(index) > 0 and hasattr(index[0], 'tzinfo') and index[0].tzinfo is not None:
            return pd.DatetimeIndex([ts.replace(tzinfo=None) for ts in index])
        return index
    
    def _extract_price_data(self, hist, symbol: str) -> Dict[str, List]:
        """
        Extract price data from yfinance DataFrame and convert to lists.
        
        Args:
            hist: DataFrame from yfinance
            symbol: Stock symbol
            
        Returns:
            Dictionary with dates, prices, opens, highs, lows, volumes
        """
        # Convert timestamps to datetime objects with timezone handling
        dates = []
        for ts in hist.index:
            # Convert pandas Timestamp to datetime
            if hasattr(ts, 'to_pydatetime'):
                dt = ts.to_pydatetime()
            elif hasattr(ts, 'timestamp'):
                dt = datetime.fromtimestamp(ts.timestamp())
            else:
                dt = ts
            
            # Convert to user's timezone if specified
            if self.timezone:
                if dt.tzinfo is not None:
                    dt = dt.astimezone(self.timezone)
                else:
                    # Assume UTC if naive
                    dt = pytz.UTC.localize(dt).astimezone(self.timezone)
            
            dates.append(dt)
        
        # Extract price data
        closes = [float(c) for c in hist['Close'].tolist()]
        opens = [float(o) for o in hist['Open'].tolist()]
        highs = [float(h) for h in hist['High'].tolist()]
        lows = [float(l) for l in hist['Low'].tolist()]
        volumes = [float(v) for v in hist['Volume'].tolist()]
        
        logger.debug(f"Fetched {len(dates)} data points for {symbol}")
        
        return {
            'symbol': symbol,
            'dates': dates,
            'prices': closes,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'volumes': volumes
        }
