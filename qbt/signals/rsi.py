import pandas as pd
import numpy as np
from .base import SignalGenerator


class RSISignal(SignalGenerator):
    """Relative Strength Index signal generator."""
    
    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        column: str = 'Close'
    ):
        """
        Initialize RSI signal generator.
        
        Args:
            period: RSI calculation period
            overbought: Overbought threshold
            oversold: Oversold threshold
            column: Price column to use for calculation
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.column = column
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI for a price series."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def transform(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI signals to the price DataFrame.
        
        Args:
            prices_df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional RSI signal columns
        """
        result_df = prices_df.copy()
        
        # Get unique symbols
        if isinstance(prices_df.index, pd.MultiIndex):
            symbols = prices_df.index.get_level_values('Symbol').unique()
            
            for symbol in symbols:
                symbol_data = prices_df.loc[pd.IndexSlice[:, symbol], :]
                
                # Calculate RSI
                rsi = self._calculate_rsi(symbol_data[self.column])
                
                # Add to result
                result_df.loc[pd.IndexSlice[:, symbol], 'RSI'] = rsi.values
                
                # Generate trading signals
                # 1 for oversold (buy signal), -1 for overbought (sell signal), 0 for neutral
                signals = np.where(rsi <= self.oversold, 1,
                                 np.where(rsi >= self.overbought, -1, 0))
                
                result_df.loc[pd.IndexSlice[:, symbol], 'RSI_Signal'] = signals
        else:
            # Single symbol case
            rsi = self._calculate_rsi(prices_df[self.column])
            result_df['RSI'] = rsi
            
            # Generate trading signals
            signals = np.where(rsi <= self.oversold, 1,
                             np.where(rsi >= self.overbought, -1, 0))
            result_df['RSI_Signal'] = signals
        
        return result_df