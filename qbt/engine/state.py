from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    quantity: int  # positive for buy, negative for sell
    order_type: str = "market"  # market, limit, etc.
    timestamp: Optional[datetime] = None


@dataclass
class Fill:
    """Represents an executed trade."""
    symbol: str
    quantity: int
    price: float
    fees: float
    slippage: float
    timestamp: datetime
    
    @property
    def total_cost(self) -> float:
        """Total cost including fees and slippage."""
        return abs(self.quantity) * self.price + self.fees + abs(self.quantity) * self.slippage


@dataclass
class PortfolioState:
    """Holds the current state of the portfolio."""
    cash: float = 100000.0
    positions: Dict[str, int] = field(default_factory=dict)
    
    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol."""
        return self.positions.get(symbol, 0)
    
    def update_position(self, symbol: str, quantity: int):
        """Update position for a symbol."""
        current_position = self.get_position(symbol)
        new_position = current_position + quantity
        
        if new_position == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_position
    
    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio equity given current prices."""
        equity = self.cash
        for symbol, position in self.positions.items():
            if symbol in prices:
                equity += position * prices[symbol]
        return equity
    
    def copy(self) -> 'PortfolioState':
        """Create a copy of the portfolio state."""
        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy()
        )