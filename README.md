# Quantitative Backtesting Framework

A modular Python framework for backtesting quantitative trading strategies with clean architecture and extensible components.

## Features

- **Modular Design**: Pluggable components for data sources, signal generators, strategies, and execution
- **Multiple Signal Generators**: EMA, MACD, RSI with easy extensibility
- **Realistic Execution**: Market orders with fees, slippage, and position management
- **Comprehensive Analytics**: Performance metrics, visualizations, and risk analysis
- **Clean Architecture**: Well-structured codebase following SOLID principles

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from qbt.examples.run_example import main
main()
```

Or run the example directly:
```bash
cd qbt/examples
python run_example.py
```

## Framework Architecture

```
qbt/
├── data/           # Data sources (Yahoo Finance, etc.)
├── signals/        # Technical indicators (EMA, MACD, RSI)
├── strategy/       # Trading strategies (CrossOver, etc.)
├── execution/      # Order execution (SimpleBroker)
├── engine/         # Core engine (Backtester, Metrics, Viz)
└── examples/       # Example implementations
```

## Core Components

### Data Source
```python
from qbt.data.yfinance_source import YFDataSource
data_source = YFDataSource()
```

### Signal Generators
```python
from qbt.signals.ema import EMASignal
from qbt.signals.macd import MACDSignal
from qbt.signals.rsi import RSISignal

ema_signal = EMASignal(short_period=12, long_period=26)
macd_signal = MACDSignal()
rsi_signal = RSISignal(period=14)
```

### Strategy
```python
from qbt.strategy.cross_over import CrossOverStrategy
strategy = CrossOverStrategy(position_size=0.3)
```

### Execution
```python
from qbt.execution.simple_broker import SimpleBroker
broker = SimpleBroker(commission=0.001, slippage=0.0005)
```

### Backtester
```python
from qbt.engine.backtester import Backtester

backtester = Backtester(
    data_source=data_source,
    signal_generators=[ema_signal],
    strategy=strategy,
    broker=broker,
    initial_cash=100000.0
)

result = backtester.run(
    universe=["AAPL", "MSFT", "GOOGL"],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

### Analytics
```python
from qbt.engine.metrics import PerformanceMetrics
from qbt.engine.viz import Visualizer

# Calculate metrics
metrics = PerformanceMetrics.calculate_metrics(result)
PerformanceMetrics.print_metrics(metrics)

# Visualize results
fig = Visualizer.plot_comprehensive_analysis(result)
```

## Example Output

The framework provides comprehensive performance metrics:

```
==================================================
PERFORMANCE METRICS
==================================================
Total Return (%)              :      45.67%
Annualized Return (%)         :      12.34%
Annualized Volatility (%)     :      18.45%
Sharpe Ratio                  :       0.67
Maximum Drawdown (%)          :     -12.34%
Max Drawdown Duration (days)  :         45
Calmar Ratio                  :       1.00
Total Trades                  :        156
Win Rate (%)                  :      55.13%
Best Day (%)                  :       4.67%
Worst Day (%)                 :      -3.89%
VaR 5% (%)                    :      -2.34%
Final Equity ($)              : $145,670.00
==================================================
```

## Extending the Framework

### Custom Signal Generator
```python
from qbt.signals.base import SignalGenerator

class MyCustomSignal(SignalGenerator):
    def transform(self, prices_df):
        # Implement your signal logic
        return prices_df
```

### Custom Strategy
```python
from qbt.strategy.base import Strategy

class MyStrategy(Strategy):
    def on_bar(self, date, slice_df, state):
        # Implement your trading logic
        return orders
```

### Custom Data Source
```python
from qbt.data.base import DataSource

class MyDataSource(DataSource):
    def get_price(self, universe, start, end, interval="1d"):
        # Implement your data fetching logic
        return dataframe
```

## Performance Visualizations

The framework generates comprehensive visualizations:

- **Equity Curve**: Portfolio value over time
- **Drawdown Analysis**: Risk assessment and recovery periods
- **Returns Distribution**: Statistical analysis of returns
- **Monthly Heatmap**: Seasonal performance patterns
- **Rolling Metrics**: Time-varying risk metrics

## Dependencies

- `yfinance`: Market data fetching
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scipy`: Statistical functions

## License

MIT License - feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Disclaimer

This framework is for educational and research purposes only. Past performance does not guarantee future results. Always perform thorough testing before deploying any trading strategy with real capital.