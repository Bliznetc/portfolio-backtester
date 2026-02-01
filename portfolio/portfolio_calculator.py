"""
Portfolio value calculation logic.
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio value at a specific date."""
    date: datetime
    total_value: float
    return_pct: float
    individual_values: Dict[str, float]  # {ticker: current_value}
    individual_returns: Dict[str, float]  # {ticker: return_pct}


class PortfolioCalculator:
    """
    Calculates portfolio values and returns over time.
    """
    
    def calculate_initial_shares(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        initial_value: float,
        start_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate initial number of shares for each ticker.
        
        Args:
            tickers: List of ticker symbols
            weights: Dictionary mapping ticker to target weight (0.0 to 1.0)
            initial_value: Total initial portfolio value
            start_prices: Dictionary mapping ticker to price on start date
            
        Returns:
            Dictionary mapping ticker to number of shares
        """
        shares = {}
        total_weight = sum(weights.values())
        
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, not 1.0. Normalizing...")
            # Normalize weights
            weights = {t: w / total_weight for t, w in weights.items()}
        
        for ticker in tickers:
            if ticker not in start_prices or start_prices[ticker] is None:
                logger.error(f"Missing start price for {ticker}")
                raise ValueError(f"Missing start price for {ticker}")
            
            weight = weights.get(ticker, 0.0)
            dollar_amount = weight * initial_value
            price = start_prices[ticker]
            
            if price <= 0:
                logger.error(f"Invalid price {price} for {ticker}")
                raise ValueError(f"Invalid price {price} for {ticker}")
            
            shares[ticker] = dollar_amount / price
            logger.debug(f"{ticker}: ${dollar_amount:.2f} / ${price:.2f} = {shares[ticker]:.4f} shares")
        
        return shares
    
    def calculate_portfolio_value(
        self,
        shares: Dict[str, float],
        prices: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate current portfolio value.
        
        Args:
            shares: Dictionary mapping ticker to number of shares (constant)
            prices: Dictionary mapping ticker to current price
            
        Returns:
            Tuple of (total_value, individual_values_dict)
        """
        total_value = 0.0
        individual_values = {}
        
        for ticker, share_count in shares.items():
            if ticker in prices and prices[ticker] is not None:
                value = share_count * prices[ticker]
                individual_values[ticker] = value
                total_value += value
            else:
                individual_values[ticker] = 0.0
        
        return total_value, individual_values
    
    def calculate_returns(
        self,
        current_value: float,
        initial_value: float,
        individual_values: Dict[str, float],
        initial_individual_values: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate percentage returns.
        
        Args:
            current_value: Current total portfolio value
            initial_value: Initial portfolio value
            individual_values: Current individual ticker values
            initial_individual_values: Initial individual ticker values
            
        Returns:
            Tuple of (total_return_pct, individual_returns_dict)
        """
        if initial_value == 0:
            return 0.0, {}
        
        total_return_pct = ((current_value - initial_value) / initial_value) * 100
        
        individual_returns = {}
        for ticker in individual_values:
            initial_val = initial_individual_values.get(ticker, 0.0)
            if initial_val > 0:
                individual_returns[ticker] = ((individual_values[ticker] - initial_val) / initial_val) * 100
            else:
                individual_returns[ticker] = 0.0
        
        return total_return_pct, individual_returns
    
    def calculate_historical_performance(
        self,
        shares: Dict[str, float],
        price_data: Dict[str, Dict],
        initial_value: float,
        start_date: datetime
    ) -> List[PortfolioSnapshot]:
        """
        Calculate portfolio value for each day in the historical data.
        
        Args:
            shares: Dictionary mapping ticker to shares (constant)
            price_data: Dictionary from PriceDataFetcher.fetch_historical_prices
            initial_value: Initial portfolio value
            start_date: Start date of the portfolio
            
        Returns:
            List of PortfolioSnapshot objects, one for each day
        """
        snapshots = []
        
        # Get all unique dates across all tickers
        all_dates = set()
        for ticker_data in price_data.values():
            all_dates.update(ticker_data['dates'])
        
        if not all_dates:
            logger.warning("No dates found in price data")
            return snapshots
        
        # Sort dates
        sorted_dates = sorted(all_dates)
        
        # Calculate initial individual values
        initial_individual_values = {}
        for ticker, share_count in shares.items():
            if ticker in price_data and price_data[ticker]['dates']:
                # Get first available price
                first_price = price_data[ticker]['prices'][0]
                initial_individual_values[ticker] = share_count * first_price
            else:
                initial_individual_values[ticker] = 0.0
        
        # Calculate portfolio value for each date
        for date_idx, date in enumerate(sorted_dates):
            # Get prices for this date for all tickers
            current_prices = {}
            for ticker, ticker_data in price_data.items():
                dates = ticker_data['dates']
                prices = ticker_data['prices']
                
                # Find price for this exact timestamp
                price = None
                for i, d in enumerate(dates):
                    if d == date:  # Exact match on full timestamp
                        price = prices[i]
                        break
                    elif d > date and i > 0:  # Closest before this timestamp
                        price = prices[i - 1]
                        break
                
                # If date is after all dates, use last price
                if price is None and dates and dates[-1] < date:
                    price = prices[-1]
                
                current_prices[ticker] = price
            
            # Calculate portfolio value
            total_value, individual_values = self.calculate_portfolio_value(shares, current_prices)
            
            # Calculate returns
            return_pct, individual_returns = self.calculate_returns(
                total_value,
                initial_value,
                individual_values,
                initial_individual_values
            )
            
            snapshot = PortfolioSnapshot(
                date=date,
                total_value=total_value,
                return_pct=return_pct,
                individual_values=individual_values,
                individual_returns=individual_returns
            )
            
            snapshots.append(snapshot)
        
        return snapshots
