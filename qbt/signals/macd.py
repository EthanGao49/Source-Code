import pandas as pd
from .base import SignalGenerator


class MACDSignal(SignalGenerator):
    """MACD (Moving Average Convergence Divergence) signal generator."""
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'Close'
    ):
        """
        Initialize MACD signal generator.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            column: Price column to use for calculation
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column
    
    def transform(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD signals to the price DataFrame.
        
        Args:
            prices_df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional MACD signal columns
        """
        result_df = prices_df.copy()
        
        # Get unique symbols
        if isinstance(prices_df.index, pd.MultiIndex):
            symbols = prices_df.index.get_level_values('Symbol').unique()
            
            for symbol in symbols:
                symbol_data = prices_df.loc[pd.IndexSlice[:, symbol], :]
                
                # Calculate MACD components
                fast_ema = symbol_data[self.column].ewm(span=self.fast_period).mean()
                slow_ema = symbol_data[self.column].ewm(span=self.slow_period).mean()
                macd_line = fast_ema - slow_ema
                signal_line = macd_line.ewm(span=self.signal_period).mean()
                histogram = macd_line - signal_line
                
                # Add to result
                result_df.loc[pd.IndexSlice[:, symbol], 'MACD'] = macd_line.values
                result_df.loc[pd.IndexSlice[:, symbol], 'MACD_Signal'] = signal_line.values
                result_df.loc[pd.IndexSlice[:, symbol], 'MACD_Histogram'] = histogram.values
                
                # Generate trading signals (1 for bullish, 0 for bearish)
                result_df.loc[pd.IndexSlice[:, symbol], 'MACD_Trading_Signal'] = (
                    (macd_line > signal_line).astype(int).values
                )
        else:
            # Single symbol case
            fast_ema = prices_df[self.column].ewm(span=self.fast_period).mean()
            slow_ema = prices_df[self.column].ewm(span=self.slow_period).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=self.signal_period).mean()
            histogram = macd_line - signal_line
            
            result_df['MACD'] = macd_line
            result_df['MACD_Signal'] = signal_line
            result_df['MACD_Histogram'] = histogram
            result_df['MACD_Trading_Signal'] = (macd_line > signal_line).astype(int)
        
        return result_df