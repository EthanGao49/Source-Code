from abc import ABC, abstractmethod
import pandas as pd


class SignalGenerator(ABC):
    """Abstract base class for signal generators."""
    
    @abstractmethod
    def transform(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trading signals to the price DataFrame.
        
        Args:
            prices_df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional signal columns
        """
        pass