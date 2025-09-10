from typing import List
from datetime import datetime
import pandas as pd
from .base import Strategy
from ..engine.state import Order, PortfolioState


class CrossOverStrategy(Strategy):
    """
    Crossover strategy based on EMA signals.
    
    Buy when short EMA crosses above long EMA.
    Sell when short EMA crosses below long EMA.
    """
    
    def __init__(
        self,
        position_size: float = 0.2,
        signal_column: str = 'EMA_Signal'
    ):
        """
        Initialize CrossOver strategy.
        
        Args:
            position_size: Fraction of portfolio to allocate per position (0.0 to 1.0)
            signal_column: Column name for the signal to follow
        """
        self.position_size = position_size
        self.signal_column = signal_column
        self.previous_signals = {}
    
    def on_bar(
        self,
        date: datetime,
        slice_df: pd.DataFrame,
        state: PortfolioState
    ) -> List[Order]:
        """
        Generate orders based on crossover signals.
        
        Args:
            date: Current date
            slice_df: Market data for current date (all symbols)
            state: Current portfolio state
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        if slice_df.empty or self.signal_column not in slice_df.columns:
            return orders
        
        # Calculate current portfolio equity
        current_prices = slice_df['Close'].to_dict() if 'Close' in slice_df.columns else {}
        total_equity = state.get_total_equity(current_prices)
        
        for symbol in slice_df.index:
            if pd.isna(slice_df.loc[symbol, self.signal_column]):
                continue
                
            current_signal = slice_df.loc[symbol, self.signal_column]
            previous_signal = self.previous_signals.get(symbol, 0)
            
            # Check for signal changes
            signal_changed = current_signal != previous_signal
            
            if signal_changed:
                current_position = state.get_position(symbol)
                current_price = current_prices.get(symbol, 0)
                
                if current_price <= 0:
                    continue
                
                # Buy signal: short EMA crossed above long EMA
                if current_signal == 1 and previous_signal == 0:
                    if current_position <= 0:  # Not currently long
                        # Close any short position first
                        if current_position < 0:
                            orders.append(Order(symbol, -current_position))
                        
                        # Calculate position size
                        position_value = total_equity * self.position_size
                        shares_to_buy = int(position_value / current_price)
                        
                        if shares_to_buy > 0 and position_value <= state.cash:
                            orders.append(Order(symbol, shares_to_buy))
                
                # Sell signal: short EMA crossed below long EMA
                elif current_signal == 0 and previous_signal == 1:
                    if current_position > 0:  # Currently long
                        orders.append(Order(symbol, -current_position))
            
            # Update previous signal
            self.previous_signals[symbol] = current_signal
        
        return orders