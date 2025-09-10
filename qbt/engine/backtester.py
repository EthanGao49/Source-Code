from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from ..data.base import DataSource
from ..signals.base import SignalGenerator
from ..strategy.base import Strategy
from ..execution.base import Broker
from .state import PortfolioState, Fill


class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Fill] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.initial_cash: float = 0.0
        self.final_equity: float = 0.0
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'Date': trade.timestamp,
                'Symbol': trade.symbol,
                'Quantity': trade.quantity,
                'Price': trade.price,
                'Fees': trade.fees,
                'Slippage': trade.slippage,
                'Total_Cost': trade.total_cost
            })
        
        df = pd.DataFrame(trades_data)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df


class Backtester:
    """
    Main backtesting engine that orchestrates data, signals, strategy, and broker.
    """
    
    def __init__(
        self,
        data_source: DataSource,
        signal_generators: List[SignalGenerator],
        strategy: Strategy,
        broker: Broker,
        initial_cash: float = 100000.0
    ):
        """
        Initialize backtester.
        
        Args:
            data_source: Data source for market data
            signal_generators: List of signal generators to apply
            strategy: Trading strategy
            broker: Order execution broker
            initial_cash: Starting cash amount
        """
        self.data_source = data_source
        self.signal_generators = signal_generators
        self.strategy = strategy
        self.broker = broker
        self.initial_cash = initial_cash
    
    def run(
        self,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> BacktestResult:
        """
        Run the backtest.
        
        Args:
            universe: List of symbols to trade
            start_date: Start date for backtest
            end_date: End date for backtest
            interval: Data interval
            
        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        result = BacktestResult()
        result.start_date = start_date
        result.end_date = end_date
        result.initial_cash = self.initial_cash
        
        # Initialize portfolio state
        state = PortfolioState(cash=self.initial_cash)
        
        # Get market data
        print(f"Fetching data for {universe} from {start_date} to {end_date}...")
        prices_df = self.data_source.get_price(universe, start_date, end_date, interval)
        
        if prices_df.empty:
            print("No data available for the specified period.")
            return result
        
        # Apply signal generators
        print("Generating signals...")
        for signal_generator in self.signal_generators:
            prices_df = signal_generator.transform(prices_df)
        
        # Get unique dates
        dates = prices_df.index.get_level_values('Date').unique().sort_values()
        
        print(f"Running backtest for {len(dates)} trading days...")
        
        # Run backtest day by day
        for i, date in enumerate(dates):
            try:
                # Get current date data
                current_data = prices_df.loc[pd.IndexSlice[date, :], :]
                
                # Reset the index to make symbols the index (remove date level)
                if current_data.index.nlevels > 1:
                    current_data = current_data.reset_index(level=0, drop=True)
                
                # Ensure we have a proper DataFrame
                if isinstance(current_data, pd.Series):
                    current_data = current_data.to_frame().T
                
                if current_data.empty:
                    continue
                
                symbol_data = current_data

                # Generate orders
                orders = self.strategy.on_bar(date, symbol_data, state)

                # Execute orders
                if orders:
                    current_prices = symbol_data['Close'].to_dict() if 'Close' in symbol_data.columns else {}
                    fills = self.broker.execute(orders, current_prices, state)
                    result.trades.extend(fills)

                # Record portfolio state
                current_prices = symbol_data['Close'].to_dict() if 'Close' in symbol_data.columns else {}
                total_equity = state.get_total_equity(current_prices)

                result.equity_curve.append({
                    'Date': date,
                    'Cash': state.cash,
                    'Equity': total_equity,
                    'Positions': len(state.positions)
                })

                result.portfolio_history.append({
                    'Date': date,
                    'State': state.copy()
                })

                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(dates)} days...")

            except Exception as e:
                print(f"Error processing date {date}: {e}")
                continue
        
        # Set final equity
        if result.equity_curve:
            result.final_equity = result.equity_curve[-1]['Equity']
        
        print(f"Backtest completed. Final equity: ${result.final_equity:,.2f}")
        print(f"Total return: {((result.final_equity / result.initial_cash - 1) * 100):.2f}%")
        print(f"Number of trades: {len(result.trades)}")
        
        return result