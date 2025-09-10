from typing import List
from datetime import datetime
import pandas as pd
from .base import Strategy
from ..engine.state import Order, PortfolioState


class BuyAndHoldStrategy(Strategy):
    """
    Buy and hold benchmark strategy.
    
    This strategy buys all securities in the universe on the first trading day
    and holds them until the end of the backtest period. It's commonly used
    as a benchmark for comparing active strategies.
    """
    
    def __init__(self, allocation_method: str = "equal_weight"):
        """
        Initialize buy and hold strategy.
        
        Args:
            allocation_method: How to allocate capital across securities
                - "equal_weight": Equal allocation to each security
                - "market_cap": Market cap weighted (requires market cap data)
        """
        self.allocation_method = allocation_method
        self.initial_purchase_made = False
        self.symbols_to_buy = set()
    
    def on_bar(
        self,
        date: datetime,
        slice_df: pd.DataFrame,
        state: PortfolioState
    ) -> List[Order]:
        """
        Generate orders for buy and hold strategy.
        
        On the first bar, buy all available securities with equal weighting.
        On subsequent bars, do nothing (hold).
        
        Args:
            date: Current date
            slice_df: Market data for current date (all symbols)
            state: Current portfolio state
            
        Returns:
            List of orders to execute (empty after initial purchase)
        """
        orders = []
        
        # Only buy on the first trading day
        if not self.initial_purchase_made:
            available_symbols = slice_df.index.tolist()
            if not available_symbols:
                return orders
            
            # Use equal weight allocation
            if self.allocation_method == "equal_weight":
                allocation_per_symbol = 1.0 / len(available_symbols)
                
                for symbol in available_symbols:
                    if 'Close' in slice_df.columns and not pd.isna(slice_df.loc[symbol, 'Close']):
                        price = slice_df.loc[symbol, 'Close']
                        
                        # Calculate how much cash to allocate to this symbol
                        cash_to_allocate = state.cash * allocation_per_symbol
                        
                        # Calculate shares to buy (round down to avoid insufficient funds)
                        if price > 0:
                            shares_to_buy = int(cash_to_allocate / price)
                            
                            if shares_to_buy > 0:
                                orders.append(Order(
                                    symbol=symbol,
                                    quantity=shares_to_buy,
                                    order_type="market"
                                ))
            
            self.initial_purchase_made = True
        
        # After initial purchase, hold (do nothing)
        return orders