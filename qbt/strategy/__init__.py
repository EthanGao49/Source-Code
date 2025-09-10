from .base import Strategy
from .cross_over import CrossOverStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .market_benchmark import (
    MarketBenchmarkStrategy,
    SP500BenchmarkStrategy, 
    NASDAQ100BenchmarkStrategy,
    Russell2000BenchmarkStrategy,
    TotalMarketBenchmarkStrategy,
    create_benchmark_strategy,
    get_benchmark_universe,
    get_available_benchmarks
)