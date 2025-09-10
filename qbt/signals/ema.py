import pandas as pd
from .base import SignalGenerator


class EMASignal(SignalGenerator):
    """Exponential Moving Average signal generator."""
    
    def __init__(self, short_period: int = 12, long_period: int = 26, column: str = 'Close'):
        """
        Initialize EMA signal generator.
        
        Args:
            short_period: Short EMA period
            long_period: Long EMA period
            column: Price column to use for calculation
        """
        self.short_period = short_period
        self.long_period = long_period
        self.column = column
    
    def transform(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA signals to the price DataFrame.
        
        Args:
            prices_df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional EMA signal columns
        """
        result_df = prices_df.copy()
        
        # Get unique symbols
        if isinstance(prices_df.index, pd.MultiIndex):
            symbols = prices_df.index.get_level_values('Symbol').unique()
            
            for symbol in symbols:
                symbol_data = prices_df.loc[pd.IndexSlice[:, symbol], :]
                
                # Calculate EMAs
                short_ema = symbol_data[self.column].ewm(span=self.short_period).mean()
                long_ema = symbol_data[self.column].ewm(span=self.long_period).mean()
                
                # Add to result
                result_df.loc[pd.IndexSlice[:, symbol], f'EMA_{self.short_period}'] = short_ema.values
                result_df.loc[pd.IndexSlice[:, symbol], f'EMA_{self.long_period}'] = long_ema.values
                
                # Generate signals
                result_df.loc[pd.IndexSlice[:, symbol], 'EMA_Signal'] = (
                    (short_ema > long_ema).astype(int).values
                )
        else:
            # Single symbol case
            short_ema = prices_df[self.column].ewm(span=self.short_period).mean()
            long_ema = prices_df[self.column].ewm(span=self.long_period).mean()
            
            result_df[f'EMA_{self.short_period}'] = short_ema
            result_df[f'EMA_{self.long_period}'] = long_ema
            result_df['EMA_Signal'] = (short_ema > long_ema).astype(int)
        
        return result_df