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
from qbt.strategy.market_benchmark import create_benchmark_strategy, get_available_benchmarks
from qbt.execution.simple_broker import SimpleBroker
from qbt.engine.backtester import Backtester
from qbt.engine.metrics import PerformanceMetrics
from qbt.engine.viz import Visualizer
from qbt.engine.summary import SummaryReport


def show_benchmark_options():
    """Display available benchmark options."""
    print("\nAvailable Benchmark Options:")
    print("=" * 40)
    benchmarks = get_available_benchmarks()
    for i, (bench_type, description) in enumerate(benchmarks.items(), 1):
        print(f"{i}. {bench_type}: {description}")
    print()


def main():
    """Run the example backtest."""
    
    print("Quantitative Backtesting Framework - Example")
    print("=" * 50)
    
    # Show available benchmarks
    show_benchmark_options()
    
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
        
        # Broker
        broker = SimpleBroker(commission=0, slippage=0)
        
        # Backtester with multiple benchmarks
        backtester = Backtester(
            data_source=data_source,
            signal_generators=signal_generators,
            strategy=strategy,
            broker=broker,
            initial_cash=initial_cash,
            benchmark_types=["SP500", "NASDAQ100"]  # Multiple benchmarks
        )
        
        print("Components initialized successfully!")
        print()
        
        # Run backtest
        print("Running backtest...")
        result = backtester.run(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            strategy_name="EMA CrossOver Strategy"
        )
        
        if result.equity_curve:
            # Calculate and display metrics for all benchmarks
            print("\nCalculating performance metrics...")
            all_metrics = PerformanceMetrics.calculate_all_benchmark_metrics(result)
            
            # Display metrics for strategy and all benchmarks
            for name, metrics in all_metrics.items():
                if 'Strategy vs' in name:
                    print(f"\n📊 {name.upper()} COMPARISON:")
                elif 'Standalone' in name:
                    print(f"\n📈 {name.upper()} METRICS:")
                else:
                    print(f"\n📊 {name.upper()}:")
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
            
            # Signal analysis plot
            fig_signals = Visualizer.plot_signals(result)
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
            
            # Generate PDF report
            print("\nGenerating PDF report...")
            try:
                # Generate report (config is automatically included in result)
                report = SummaryReport(result)
                pdf_filename = report.generate_pdf()
                print(f"PDF report saved as: {pdf_filename}")
                
            except Exception as e:
                print(f"Error generating PDF report: {e}")
                import traceback
                traceback.print_exc()
            
            print("\nBacktest completed successfully!")
            
        else:
            print("No results to display. Check your data and date range.")
    
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def multiple_benchmarks_example():
    """Run example with multiple benchmarks including market indices and custom strategies."""
    
    print("\nMultiple Benchmarks Example")
    print("=" * 40)
    print("Comparing strategy against NASDAQ100 ETF and Buy & Hold custom strategy")
    
    # Configuration
    universe = ["AAPL", "MSFT", "GOOGL"]
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_cash = 100000.0
    
    try:
        # Components
        data_source = YFDataSource()
        ema_signal = EMASignal(short_period=10, long_period=30)
        strategy = CrossOverStrategy(position_size=0.4)
        broker = SimpleBroker(commission=0.001, slippage=0.0005)
        
        # Create multiple benchmarks: market benchmark + custom strategy
        buy_and_hold_strategy = BuyAndHoldStrategy(allocation_method="equal_weight")
        
        benchmark_strategies = {
            "Buy_and_Hold": buy_and_hold_strategy
        }
        
        # Backtester with both market benchmark and custom benchmark
        backtester = Backtester(
            data_source=data_source,
            signal_generators=[ema_signal],
            strategy=strategy,
            broker=broker,
            initial_cash=initial_cash,
            benchmark_types=["NASDAQ100"],  # Market benchmark
            benchmark_strategies=benchmark_strategies  # Custom benchmark
        )
        
        # Run backtest
        result = backtester.run(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            strategy_name="EMA Strategy with Multiple Benchmarks"
        )
        
        if result.equity_curve:
            # Calculate metrics for all benchmarks
            from qbt.engine.metrics import PerformanceMetrics
            all_metrics = PerformanceMetrics.calculate_all_benchmark_metrics(result)
            
            print(f"\n{'='*60}")
            print("MULTIPLE BENCHMARKS COMPARISON")
            print(f"{'='*60}")
            
            for name, metrics in all_metrics.items():
                if name == 'Strategy':
                    print(f"\n📈 {name.upper()} PERFORMANCE:")
                else:
                    print(f"\n📊 {name.upper()} BENCHMARK:")
                
                print(f"  Total Return: {metrics.get('Total Return (%)', 0):.2f}%")
                print(f"  Annualized Return: {metrics.get('Annualized Return (%)', 0):.2f}%")
                print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('Maximum Drawdown (%)', 0):.2f}%")
                
                if 'Alpha (%)' in metrics:
                    print(f"  Alpha: {metrics['Alpha (%)']:.2f}%")
                    print(f"  Beta: {metrics.get('Beta', 0):.2f}")
                    print(f"  Information Ratio: {metrics.get('Information Ratio', 0):.2f}")
            
            # Generate comprehensive plot with multiple benchmarks
            fig = Visualizer.plot_comprehensive_analysis(result)
            plt.show()
            
        else:
            print("No results generated.")
            
    except Exception as e:
        print(f"Multiple benchmarks example failed: {e}")
        import traceback
        traceback.print_exc()


def benchmark_comparison_example():
    """Run example comparing different benchmarks."""
    
    print("\nBenchmark Comparison Example")
    print("=" * 40)
    print("Comparing strategy against S&P 500 and NASDAQ 100 benchmarks")
    
    # Configuration
    universe = ["AAPL", "MSFT"]
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_cash = 50000.0
    
    benchmarks_to_test = ["SP500", "NASDAQ100"]
    results = {}
    
    for benchmark_type in benchmarks_to_test:
        print(f"\nRunning with {benchmark_type} benchmark...")
        
        try:
            # Components
            data_source = YFDataSource()
            ema_signal = EMASignal(short_period=10, long_period=30)
            strategy = CrossOverStrategy(position_size=0.5)
            broker = SimpleBroker(commission=0.001, slippage=0.0005)
            
            # Backtester
            backtester = Backtester(
                data_source=data_source,
                signal_generators=[ema_signal],
                strategy=strategy,
                broker=broker,
                initial_cash=initial_cash,
                benchmark_type=benchmark_type
            )
            
            # Run backtest
            result = backtester.run(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                strategy_name=f"EMA Strategy vs {benchmark_type}"
            )
            
            if result.equity_curve:
                metrics = PerformanceMetrics.calculate_metrics(result)
                results[benchmark_type] = {
                    'strategy_return': metrics['Total Return (%)'],
                    'benchmark_return': metrics.get('Benchmark Total Return (%)', 0),
                    'alpha': metrics.get('Alpha (%)', 0),
                    'sharpe': metrics['Sharpe Ratio'],
                    'max_drawdown': metrics['Maximum Drawdown (%)']
                }
                
                print(f"Strategy Return: {metrics['Total Return (%)']:.2f}%")
                print(f"Benchmark Return: {metrics.get('Benchmark Total Return (%)', 0):.2f}%")
                print(f"Alpha: {metrics.get('Alpha (%)', 0):.2f}%")
                
        except Exception as e:
            print(f"Error with {benchmark_type} benchmark: {e}")
    
    # Summary comparison
    if results:
        print(f"\n{'='*50}")
        print("BENCHMARK COMPARISON SUMMARY")
        print(f"{'='*50}")
        for benchmark, metrics in results.items():
            print(f"\n{benchmark} Benchmark:")
            print(f"  Strategy Return: {metrics['strategy_return']:.2f}%")
            print(f"  Benchmark Return: {metrics['benchmark_return']:.2f}%")
            print(f"  Alpha: {metrics['alpha']:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")


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
        broker = SimpleBroker(commission=0.0005, slippage=0.0002)  # Lower costs
        
        # Backtester with NASDAQ 100 benchmark (since we're testing AAPL)
        backtester = Backtester(
            data_source=data_source,
            signal_generators=[ema_signal],
            strategy=strategy,
            broker=broker,
            initial_cash=initial_cash,
            benchmark_type="NASDAQ100"  # Use NASDAQ 100 as benchmark
        )
        
        # Run backtest
        result = backtester.run(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            strategy_name="Fast EMA CrossOver Strategy"
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
            
            # Simple signals plot
            fig_signals = Visualizer.plot_signals(result, title="AAPL Trading Signals")
            plt.show()
            
            # Generate simple PDF report
            print("\nGenerating PDF report...")
            try:
                report = SummaryReport(result)
                pdf_filename = report.generate_pdf("simple_backtest_report.pdf")
                print(f"Simple PDF report saved as: {pdf_filename}")
                
            except Exception as e:
                print(f"Error generating PDF report: {e}")
            
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
    
    # Optionally run multiple benchmarks example
    user_input = input("\nRun multiple benchmarks example? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        multiple_benchmarks_example()
    
    # Optionally run benchmark comparison example
    user_input = input("\nRun benchmark comparison example? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        benchmark_comparison_example()
    
    sys.exit(exit_code)