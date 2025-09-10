#!/usr/bin/env python3
"""
Example script demonstrating the quantitative backtesting framework.

This script:
1. Downloads OHLCV data for AAPL, MSFT, and GOOGL
2. Generates EMA signals
3. Runs a CrossOver strategy
4. Displays performance metrics and plots
"""

import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the parent directory to the path to import qbt
sys.path.append('..')

from qbt.data.yfinance_source import YFDataSource
from qbt.signals.ema import EMASignal
from qbt.strategy.cross_over import CrossOverStrategy
from qbt.strategy.buy_and_hold import BuyAndHoldStrategy
from qbt.execution.simple_broker import SimpleBroker
from qbt.engine.backtester import Backtester
from qbt.engine.metrics import PerformanceMetrics
from qbt.engine.viz import Visualizer


def main():
    """Run the example backtest."""
    
    print("Quantitative Backtesting Framework - Example")
    print("=" * 50)
    
    # Configuration
    universe = ["AAPL", "MSFT", "GOOGL"]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_cash = 100000.0
    
    print(f"Universe: {universe}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print()
    
    try:
        # Initialize components
        print("Initializing components...")
        
        # Data source
        data_source = YFDataSource()
        
        # Signal generators
        ema_signal = EMASignal(short_period=12, long_period=26)
        signal_generators = [ema_signal]
        
        # Strategy
        strategy = CrossOverStrategy(position_size=0.3)  # 30% per position
        
        # Benchmark strategy (buy and hold)
        benchmark_strategy = BuyAndHoldStrategy(allocation_method="equal_weight")
        
        # Broker
        broker = SimpleBroker(commission=0.001, slippage=0)
        
        # Backtester
        backtester = Backtester(
            data_source=data_source,
            signal_generators=signal_generators,
            strategy=strategy,
            broker=broker,
            initial_cash=initial_cash,
            benchmark_strategy=benchmark_strategy
        )
        
        print("Components initialized successfully!")
        print()
        
        # Run backtest
        print("Running backtest...")
        result = backtester.run(
            universe=universe,
            start_date=start_date,
            end_date=end_date
        )
        
        if result.equity_curve:
            # Calculate and display metrics
            print("\nCalculating performance metrics...")
            metrics = PerformanceMetrics.calculate_metrics(result)
            PerformanceMetrics.print_metrics(metrics)
            
            # Display trade summary
            trades_df = result.get_trades_dataframe()
            if not trades_df.empty:
                print(f"\nTrade Summary:")
                print(f"Total Trades: {len(trades_df)}")
                print(f"Symbols Traded: {trades_df['Symbol'].unique()}")
                print("\nFirst 10 trades:")
                print(trades_df.head(10).to_string())
            
            # Create visualizations
            print("\nGenerating visualizations...")
            
            # Comprehensive analysis plot
            fig = Visualizer.plot_comprehensive_analysis(result)
            plt.show()
            
            # Individual plots
            fig_equity = Visualizer.plot_equity_curve(result)
            plt.show()
            
            fig_drawdown = Visualizer.plot_drawdown(result)
            plt.show()
            
            fig_returns = Visualizer.plot_returns_distribution(result)
            plt.show()
            
            # Monthly returns heatmap (if enough data)
            try:
                fig_monthly = Visualizer.plot_monthly_returns_heatmap(result)
                plt.show()
            except Exception as e:
                print(f"Could not generate monthly returns heatmap: {e}")
            
            print("\nBacktest completed successfully!")
            
        else:
            print("No results to display. Check your data and date range.")
    
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def simple_example():
    """Run a simpler example with just one stock for testing."""
    
    print("\nRunning Simple Example (AAPL only)")
    print("=" * 40)
    
    # Configuration for simple test
    universe = ["AAPL"]
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 6, 30)
    initial_cash = 50000.0
    
    try:
        # Initialize components
        data_source = YFDataSource()
        ema_signal = EMASignal(short_period=5, long_period=20)  # Faster signals
        strategy = CrossOverStrategy(position_size=0.8)  # More aggressive
        benchmark_strategy = BuyAndHoldStrategy(allocation_method="equal_weight")
        broker = SimpleBroker(commission=0.0005, slippage=0.0002)  # Lower costs
        
        # Backtester
        backtester = Backtester(
            data_source=data_source,
            signal_generators=[ema_signal],
            strategy=strategy,
            broker=broker,
            initial_cash=initial_cash,
            benchmark_strategy=benchmark_strategy
        )
        
        # Run backtest
        result = backtester.run(
            universe=universe,
            start_date=start_date,
            end_date=end_date
        )
        
        if result.equity_curve:
            # Quick metrics
            metrics = PerformanceMetrics.calculate_metrics(result)
            print(f"\nQuick Results:")
            print(f"Total Return: {metrics['Total Return (%)']:.2f}%")
            print(f"Annualized Return: {metrics['Annualized Return (%)']:.2f}%")
            print(f"Max Drawdown: {metrics['Maximum Drawdown (%)']:.2f}%")
            print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
            print(f"Total Trades: {metrics['Total Trades']}")
            
            # Simple equity plot
            fig = Visualizer.plot_equity_curve(result, title="Simple AAPL Backtest")
            plt.show()
            
        else:
            print("No results generated.")
    
    except Exception as e:
        print(f"Simple example failed: {e}")


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import yfinance
        import pandas
        import numpy
        import matplotlib
        from scipy import stats
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install yfinance pandas numpy matplotlib scipy")
        sys.exit(1)
    
    # Run main example
    exit_code = main()
    
    # Optionally run simple example
    user_input = input("\nRun simple example too? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        simple_example()
    
    sys.exit(exit_code)