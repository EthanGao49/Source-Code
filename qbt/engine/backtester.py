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
        # Benchmark data (legacy single benchmark support)
        self.benchmark_equity_curve: List[Dict[str, Any]] = []
        self.benchmark_trades: List[Fill] = []
        self.benchmark_final_equity: float = 0.0
        # Multiple benchmarks support
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        # Configuration data
        self.config: Dict[str, Any] = {}
        # Market data with signals
        self.market_data: Optional[pd.DataFrame] = None
        
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
    
    def get_benchmark_dataframe(self, benchmark_name: str = None) -> pd.DataFrame:
        """Convert benchmark equity curve to DataFrame."""
        # Legacy support for single benchmark
        if benchmark_name is None:
            if not self.benchmark_equity_curve:
                return pd.DataFrame()
            
            df = pd.DataFrame(self.benchmark_equity_curve)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        
        # Multiple benchmarks support
        if benchmark_name not in self.benchmarks:
            return pd.DataFrame()
        
        equity_curve = self.benchmarks[benchmark_name].get('equity_curve', [])
        if not equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(equity_curve)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def get_benchmark_names(self) -> List[str]:
        """Get list of benchmark names."""
        return list(self.benchmarks.keys())
    
    def add_benchmark_result(self, name: str, equity_curve: List[Dict[str, Any]], 
                           trades: List[Fill], final_equity: float):
        """Add benchmark results."""
        self.benchmarks[name] = {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_equity': final_equity
        }


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
        initial_cash: float = 100000.0,
        benchmark_strategy: Optional[Strategy] = None,
        benchmark_type: str = None,
        benchmark_strategies: Dict[str, Strategy] = None,
        benchmark_types: List[str] = None
    ):
        """
        Initialize backtester.
        
        Args:
            data_source: Data source for market data
            signal_generators: List of signal generators to apply
            strategy: Trading strategy
            broker: Order execution broker
            initial_cash: Starting cash amount
            benchmark_strategy: Optional single benchmark strategy (legacy)
            benchmark_type: Optional single benchmark type (legacy)
            benchmark_strategies: Dict of named benchmark strategies
            benchmark_types: List of benchmark types to create
        """
        self.data_source = data_source
        self.signal_generators = signal_generators
        self.strategy = strategy
        self.broker = broker
        self.initial_cash = initial_cash
        
        # Handle multiple benchmarks configuration
        self.benchmark_strategies = {}
        
        # Legacy single benchmark support
        if benchmark_strategy is not None:
            self.benchmark_strategies['Benchmark'] = benchmark_strategy
            self.benchmark_strategy = benchmark_strategy  # For backward compatibility
        elif benchmark_type is not None:
            from ..strategy.market_benchmark import create_benchmark_strategy
            strategy_obj = create_benchmark_strategy(benchmark_type)
            self.benchmark_strategies[benchmark_type] = strategy_obj
            self.benchmark_strategy = strategy_obj  # For backward compatibility
        
        # Multiple benchmarks from strategies dict
        if benchmark_strategies:
            self.benchmark_strategies.update(benchmark_strategies)
        
        # Multiple benchmarks from types list
        if benchmark_types:
            from ..strategy.market_benchmark import create_benchmark_strategy
            for bench_type in benchmark_types:
                if bench_type not in self.benchmark_strategies:
                    self.benchmark_strategies[bench_type] = create_benchmark_strategy(bench_type)
        
        # Set legacy benchmark_strategy to first one if available
        if not hasattr(self, 'benchmark_strategy') and self.benchmark_strategies:
            self.benchmark_strategy = next(iter(self.benchmark_strategies.values()))
    
    def run(
        self,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        strategy_name: str = None
    ) -> BacktestResult:
        """
        Run the backtest.
        
        Args:
            universe: List of symbols to trade
            start_date: Start date for backtest
            end_date: End date for backtest
            interval: Data interval
            strategy_name: Optional name for the strategy (for reporting)
            
        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        result = BacktestResult()
        result.start_date = start_date
        result.end_date = end_date
        result.initial_cash = self.initial_cash
        
        # Populate configuration automatically
        result.config = self._create_config(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            interval=interval,
            strategy_name=strategy_name
        )
        
        # Initialize portfolio state
        state = PortfolioState(cash=self.initial_cash)
        
        # Initialize benchmark states for multiple benchmarks
        benchmark_states = {}
        for bench_name in self.benchmark_strategies.keys():
            benchmark_states[bench_name] = PortfolioState(cash=self.initial_cash)
        
        # Legacy benchmark state
        benchmark_state = None
        if self.benchmark_strategy:
            benchmark_state = PortfolioState(cash=self.initial_cash)
        
        # Expand universe to include benchmark symbols if needed
        expanded_universe = universe.copy()
        
        for bench_name, bench_strategy in self.benchmark_strategies.items():
            if hasattr(bench_strategy, 'benchmark_symbol'):
                benchmark_symbol = bench_strategy.benchmark_symbol
                if benchmark_symbol not in expanded_universe:
                    expanded_universe.append(benchmark_symbol)
        
        # Legacy support
        if self.benchmark_strategy and hasattr(self.benchmark_strategy, 'benchmark_symbol'):
            benchmark_symbol = self.benchmark_strategy.benchmark_symbol
            if benchmark_symbol not in expanded_universe:
                expanded_universe.append(benchmark_symbol)
                print(f"Added benchmark symbol {benchmark_symbol} to universe")
        
        print(f"DEBUG: Final universe: {expanded_universe}")
        
        # Get market data
        print(f"Fetching data for {expanded_universe} from {start_date} to {end_date}...")
        prices_df = self.data_source.get_price(expanded_universe, start_date, end_date, interval)
        
        if prices_df.empty:
            print("No data available for the specified period.")
            return result
        
        # Apply signal generators
        print("Generating signals...")
        for signal_generator in self.signal_generators:
            prices_df = signal_generator.transform(prices_df)
        
        # Store market data with signals for visualization
        result.market_data = prices_df.copy()
        
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

                # Filter data for strategy to only include original universe symbols
                strategy_data = symbol_data.loc[symbol_data.index.isin(universe)]

                # Generate orders
                orders = self.strategy.on_bar(date, strategy_data, state)

                # Execute orders
                if orders:
                    current_prices = symbol_data['Close'].to_dict() if 'Close' in symbol_data.columns else {}
                    fills = self.broker.execute(orders, current_prices, state, date)
                    result.trades.extend(fills)

                # Run multiple benchmark strategies
                for bench_name, bench_strategy in self.benchmark_strategies.items():
                    bench_state = benchmark_states[bench_name]
                    benchmark_orders = bench_strategy.on_bar(date, symbol_data, bench_state)
                    if benchmark_orders:
                        current_prices = symbol_data['Close'].to_dict() if 'Close' in symbol_data.columns else {}
                        benchmark_fills = self.broker.execute(benchmark_orders, current_prices, bench_state, date)
                        
                        # Store in benchmarks dict
                        if bench_name not in result.benchmarks:
                            result.benchmarks[bench_name] = {'equity_curve': [], 'trades': [], 'final_equity': 0.0}
                        result.benchmarks[bench_name]['trades'].extend(benchmark_fills)

                # Legacy benchmark strategy support
                if self.benchmark_strategy and benchmark_state:
                    benchmark_orders = self.benchmark_strategy.on_bar(date, symbol_data, benchmark_state)
                    if benchmark_orders:
                        current_prices = symbol_data['Close'].to_dict() if 'Close' in symbol_data.columns else {}
                        benchmark_fills = self.broker.execute(benchmark_orders, current_prices, benchmark_state, date)
                        result.benchmark_trades.extend(benchmark_fills)

                # Record portfolio state
                current_prices = symbol_data['Close'].to_dict() if 'Close' in symbol_data.columns else {}
                total_equity = state.get_total_equity(current_prices)

                result.equity_curve.append({
                    'Date': date,
                    'Cash': state.cash,
                    'Equity': total_equity,
                    'Positions': len(state.positions)
                })

                # Record multiple benchmark states
                for bench_name, bench_state in benchmark_states.items():
                    benchmark_equity = bench_state.get_total_equity(current_prices)
                    if bench_name not in result.benchmarks:
                        result.benchmarks[bench_name] = {'equity_curve': [], 'trades': [], 'final_equity': 0.0}
                    
                    result.benchmarks[bench_name]['equity_curve'].append({
                        'Date': date,
                        'Cash': bench_state.cash,
                        'Equity': benchmark_equity,
                        'Positions': len(bench_state.positions)
                    })

                # Legacy benchmark state recording
                if benchmark_state:
                    benchmark_equity = benchmark_state.get_total_equity(current_prices)
                    result.benchmark_equity_curve.append({
                        'Date': date,
                        'Cash': benchmark_state.cash,
                        'Equity': benchmark_equity,
                        'Positions': len(benchmark_state.positions)
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
        
        # Set final equity for multiple benchmarks
        for bench_name in result.benchmarks.keys():
            if result.benchmarks[bench_name]['equity_curve']:
                result.benchmarks[bench_name]['final_equity'] = result.benchmarks[bench_name]['equity_curve'][-1]['Equity']
            else:
                print(f"WARNING: {bench_name} has no equity curve data")
                result.benchmarks[bench_name]['final_equity'] = result.initial_cash  # No change from initial
        
        # Legacy benchmark final equity
        if result.benchmark_equity_curve:
            result.benchmark_final_equity = result.benchmark_equity_curve[-1]['Equity']
        
        print(f"Backtest completed. Final equity: ${result.final_equity:,.2f}")
        print(f"Total return: {((result.final_equity / result.initial_cash - 1) * 100):.2f}%")
        print(f"Number of trades: {len(result.trades)}")
        
        # Print multiple benchmark results
        for bench_name, bench_data in result.benchmarks.items():
            if bench_data['equity_curve']:
                benchmark_return = ((bench_data['final_equity'] / result.initial_cash - 1) * 100)
                alpha = ((result.final_equity / result.initial_cash - 1) - (bench_data['final_equity'] / result.initial_cash - 1)) * 100
                print(f"{bench_name} return: {benchmark_return:.2f}%")
                print(f"Alpha vs {bench_name}: {alpha:.2f}%")
                print(f"DEBUG: {bench_name} final equity: ${bench_data['final_equity']:,.2f}, trades: {len(bench_data['trades'])}")
            else:
                print(f"WARNING: No equity curve data for {bench_name}")
        
        # Legacy benchmark output
        if result.benchmark_equity_curve:
            benchmark_return = ((result.benchmark_final_equity / result.initial_cash - 1) * 100)
            print(f"Benchmark (Buy & Hold) return: {benchmark_return:.2f}%")
            alpha = ((result.final_equity / result.initial_cash - 1) - (result.benchmark_final_equity / result.initial_cash - 1)) * 100
            print(f"Alpha vs Benchmark: {alpha:.2f}%")
        
        return result
    
    def _create_config(
        self,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_cash: float,
        interval: str,
        strategy_name: str = None
    ) -> Dict[str, Any]:
        """
        Create configuration dictionary automatically.
        
        Args:
            universe: List of symbols
            start_date: Backtest start date
            end_date: Backtest end date
            initial_cash: Starting capital
            interval: Data interval
            strategy_name: Optional strategy name
            
        Returns:
            Configuration dictionary
        """
        # Extract strategy information
        strategy_config = {
            'name': strategy_name or self.strategy.__class__.__name__,
            'class': self.strategy.__class__.__name__,
        }
        
        # Add strategy-specific configuration if available
        if hasattr(self.strategy, 'position_size'):
            strategy_config['position_size'] = self.strategy.position_size
        if hasattr(self.strategy, 'signal_column'):
            strategy_config['signal_column'] = self.strategy.signal_column
        if hasattr(self.strategy, 'allocation_method'):
            strategy_config['allocation_method'] = self.strategy.allocation_method
        
        # Extract signal generator information
        signals_config = []
        for signal_gen in self.signal_generators:
            signal_info = signal_gen.__class__.__name__
            if hasattr(signal_gen, 'short_period') and hasattr(signal_gen, 'long_period'):
                signal_info += f" ({signal_gen.short_period}, {signal_gen.long_period} periods)"
            elif hasattr(signal_gen, 'period'):
                signal_info += f" ({signal_gen.period} periods)"
            signals_config.append(signal_info)
        
        # Extract broker configuration
        broker_config = {
            'class': self.broker.__class__.__name__,
        }
        if hasattr(self.broker, 'commission'):
            broker_config['commission'] = self.broker.commission
        if hasattr(self.broker, 'slippage'):
            broker_config['slippage'] = self.broker.slippage
        
        # Extract benchmark configuration if available
        benchmark_config = None
        if self.benchmark_strategy:
            benchmark_config = {
                'name': self.benchmark_strategy.__class__.__name__,
                'class': self.benchmark_strategy.__class__.__name__,
            }
            if hasattr(self.benchmark_strategy, 'allocation_method'):
                benchmark_config['allocation_method'] = self.benchmark_strategy.allocation_method
        
        config = {
            'universe': universe,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': initial_cash,
            'interval': interval,
            'strategy': strategy_config,
            'signals': signals_config,
            'broker': broker_config
        }
        
        if benchmark_config:
            config['benchmark'] = benchmark_config
        
        return config