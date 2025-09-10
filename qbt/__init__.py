"""
Quantitative Backtesting (QBT) - A Python library for backtesting trading strategies.
"""

__version__ = "0.1.0"
__author__ = "QBT Development Team"

from .engine.backtester import Backtester
from .engine.state import PortfolioState
from .engine.metrics import PerformanceMetrics
from .data.yfinance_source import YFDataSource
from .execution.simple_broker import SimpleBroker
from .strategy.base import Strategy
from .strategy.cross_over import CrossOverStrategy
from .signals.base import SignalGenerator
from .signals.macd import MACDSignal
from .signals.ema import EMASignal
from .signals.rsi import RSISignal


__all__ = [
    "Backtester",
    "PortfolioState", 
    "PerformanceMetrics",
    "YFDataSource",
    "SimpleBroker",
    "Strategy",
    "CrossOverStrategy",
    "SignalGenerator",
    "MACDSignal",
    "EMASignal", 
    "RSISignal"
]