from typing import List
from datetime import datetime
import pandas as pd
from .base import Strategy
from ..engine.state import Order, PortfolioState


class MarketBenchmarkStrategy(Strategy):
    """
    Market benchmark strategy for comparing against major indices.
    
    Supports S&P 500 (SPY), NASDAQ 100 (QQQ), and other market ETFs.
    This strategy buys and holds the specified market index ETF.
    """
    
    def __init__(self, benchmark_type: str = "SP500"):
        """
        Initialize market benchmark strategy.
        
        Args:
            benchmark_type: Type of benchmark
                - "SP500": S&P 500 via SPY ETF
                - "NASDAQ100": NASDAQ 100 via QQQ ETF
                - "RUSSELL2000": Russell 2000 via IWM ETF
                - "TOTAL_MARKET": Total Stock Market via VTI ETF
        """
        self.benchmark_type = benchmark_type.upper()
        self.benchmark_symbol = self._get_benchmark_symbol()
        self.initial_purchase_made = False
        
    def _get_benchmark_symbol(self) -> str:
        """Get the ETF symbol for the benchmark type."""
        benchmark_symbols = {
            "SP500": "SPY",
            "NASDAQ100": "QQQ", 
            "RUSSELL2000": "IWM",
            "TOTAL_MARKET": "VTI",
            "DOW": "DIA",
            "EMERGING_MARKETS": "EEM",
            "INTERNATIONAL": "EFA"
        }
        
        return benchmark_symbols.get(self.benchmark_type, "SPY")
    
    def on_bar(
        self,
        date: datetime,
        slice_df: pd.DataFrame,
        state: PortfolioState
    ) -> List[Order]:
        """
        Generate orders for market benchmark strategy.
        
        On the first bar, buy the benchmark ETF with all available cash.
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
            # Check if our benchmark symbol is in the current data
            if self.benchmark_symbol in slice_df.index:
                if 'Close' in slice_df.columns and not pd.isna(slice_df.loc[self.benchmark_symbol, 'Close']):
                    price = slice_df.loc[self.benchmark_symbol, 'Close']
                    
                    if price > 0:
                        # Buy as many shares as possible with available cash
                        shares_to_buy = int(state.cash / price)
                        
                        if shares_to_buy > 0:
                            orders.append(Order(
                                symbol=self.benchmark_symbol,
                                quantity=shares_to_buy,
                                order_type="market",
                                timestamp=date
                            ))
            
            self.initial_purchase_made = True
        
        # After initial purchase, hold (do nothing)
        return orders


class SP500BenchmarkStrategy(MarketBenchmarkStrategy):
    """S&P 500 benchmark strategy using SPY ETF."""
    
    def __init__(self):
        super().__init__("SP500")


class NASDAQ100BenchmarkStrategy(MarketBenchmarkStrategy):
    """NASDAQ 100 benchmark strategy using QQQ ETF."""
    
    def __init__(self):
        super().__init__("NASDAQ100")


class Russell2000BenchmarkStrategy(MarketBenchmarkStrategy):
    """Russell 2000 benchmark strategy using IWM ETF."""
    
    def __init__(self):
        super().__init__("RUSSELL2000")


class TotalMarketBenchmarkStrategy(MarketBenchmarkStrategy):
    """Total Stock Market benchmark strategy using VTI ETF."""
    
    def __init__(self):
        super().__init__("TOTAL_MARKET")


def create_benchmark_strategy(benchmark_type: str) -> MarketBenchmarkStrategy:
    """
    Factory function to create benchmark strategies.
    
    Args:
        benchmark_type: Type of benchmark to create
        
    Returns:
        Appropriate benchmark strategy instance
    """
    benchmark_type = benchmark_type.upper()
    
    if benchmark_type == "SP500":
        return SP500BenchmarkStrategy()
    elif benchmark_type == "NASDAQ100":
        return NASDAQ100BenchmarkStrategy()
    elif benchmark_type == "RUSSELL2000":
        return Russell2000BenchmarkStrategy()
    elif benchmark_type == "TOTAL_MARKET":
        return TotalMarketBenchmarkStrategy()
    else:
        # Default to S&P 500
        return SP500BenchmarkStrategy()


def get_benchmark_universe(benchmark_type: str, user_universe: List[str] = None) -> List[str]:
    """
    Get the complete universe including benchmark symbol.
    
    Args:
        benchmark_type: Type of benchmark
        user_universe: User's trading universe
        
    Returns:
        Combined universe with benchmark symbol
    """
    strategy = create_benchmark_strategy(benchmark_type)
    benchmark_symbol = strategy.benchmark_symbol
    
    if user_universe is None:
        return [benchmark_symbol]
    
    # Add benchmark symbol if not already present
    universe = list(user_universe)
    if benchmark_symbol not in universe:
        universe.append(benchmark_symbol)
    
    return universe


def get_available_benchmarks() -> dict:
    """
    Get dictionary of available benchmark types and their descriptions.
    
    Returns:
        Dictionary mapping benchmark type to description
    """
    return {
        "SP500": "S&P 500 (SPY ETF) - Large cap U.S. stocks",
        "NASDAQ100": "NASDAQ 100 (QQQ ETF) - Large cap tech stocks", 
        "RUSSELL2000": "Russell 2000 (IWM ETF) - Small cap U.S. stocks",
        "TOTAL_MARKET": "Total Stock Market (VTI ETF) - Entire U.S. stock market",
        "DOW": "Dow Jones (DIA ETF) - 30 large U.S. companies",
        "EMERGING_MARKETS": "Emerging Markets (EEM ETF) - Emerging market stocks",
        "INTERNATIONAL": "International Developed (EFA ETF) - International developed markets"
    }