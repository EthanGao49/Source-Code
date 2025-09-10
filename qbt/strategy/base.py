from abc import ABC, abstractmethod
from typing import List
from datetime import datetime
import pandas as pd
from ..engine.state import Order, PortfolioState


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def on_bar(
        self,
        date: datetime,
        slice_df: pd.DataFrame,
        state: PortfolioState
    ) -> List[Order]:
        """
        Generate orders based on current market data and portfolio state.
        
        Args:
            date: Current date
            slice_df: Market data for current date (all symbols)
            state: Current portfolio state
            
        Returns:
            List of orders to execute
        """
        pass