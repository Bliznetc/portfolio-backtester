"""
Portfolio backtesting service for analyzing investment portfolio performance.

This module provides tools for:
- Creating portfolios with custom ticker weights
- Backtesting portfolio performance over different time periods
- Interactive visualization of portfolio returns
"""

from .core.portfolio_backtester import PortfolioBacktester
from .core.data_fetcher import PriceDataFetcher
from .core.portfolio_calculator import PortfolioCalculator

__all__ = ['PortfolioBacktester', 'PriceDataFetcher', 'PortfolioCalculator']
