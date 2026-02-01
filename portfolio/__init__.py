"""
Portfolio backtesting service for analyzing investment portfolio performance.

This module provides tools for:
- Creating portfolios with custom ticker weights
- Backtesting portfolio performance over different time periods
- Interactive visualization of portfolio returns
"""

from .portfolio_backtester import PortfolioBacktester
from .data_fetcher import PriceDataFetcher
from .portfolio_calculator import PortfolioCalculator

__all__ = ['PortfolioBacktester', 'PriceDataFetcher', 'PortfolioCalculator']
