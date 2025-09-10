from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
import pandas as pd


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def get_price(
        self,
        universe: List[str],
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for given universe and time range.
        
        Args:
            universe: List of symbols to fetch
            start: Start date
            end: End date
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with MultiIndex [date, symbol] and OHLCV columns
        """
        pass