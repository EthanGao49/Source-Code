from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime
import pandas as pd
from ..engine.state import Order, Fill, PortfolioState


class Broker(ABC):
    """Abstract base class for order execution."""
    
    @abstractmethod
    def execute(
        self,
        orders: List[Order],
        prices_today: Dict[str, float],
        state: PortfolioState
    ) -> List[Fill]:
        """
        Execute orders and return fills.
        
        Args:
            orders: List of orders to execute
            prices_today: Current market prices by symbol
            state: Current portfolio state
            
        Returns:
            List of executed fills
        """
        pass