from typing import List, Dict
from datetime import datetime
from .base import Broker
from ..engine.state import Order, Fill, PortfolioState


class SimpleBroker(Broker):
    """
    Simple broker implementation with market orders, fees, and slippage.
    """
    
    def __init__(
        self,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005   # 0.05% slippage
    ):
        """
        Initialize simple broker.
        
        Args:
            commission: Commission rate (fraction of trade value)
            slippage: Slippage rate (fraction of price)
        """
        self.commission = commission
        self.slippage = slippage
    
    def execute(
        self,
        orders: List[Order],
        prices_today: Dict[str, float],
        state: PortfolioState
    ) -> List[Fill]:
        """
        Execute market orders with fees and slippage.
        
        Args:
            orders: List of orders to execute
            prices_today: Current market prices by symbol
            state: Current portfolio state
            
        Returns:
            List of executed fills
        """
        fills = []
        
        for order in orders:
            if order.symbol not in prices_today:
                continue
                
            price = prices_today[order.symbol]
            if price <= 0:
                continue
            
            # Apply slippage (negative for buys, positive for sells)
            slippage_amount = price * self.slippage * (1 if order.quantity > 0 else -1)
            execution_price = price + slippage_amount
            
            # Calculate fees
            trade_value = abs(order.quantity) * execution_price
            fees = trade_value * self.commission
            
            # Check if we have enough cash for buy orders
            if order.quantity > 0:  # Buy order
                total_cost = trade_value + fees + abs(order.quantity) * abs(slippage_amount)
                if total_cost > state.cash:
                    # Adjust order size to available cash
                    max_shares = int(state.cash / (execution_price + fees/abs(order.quantity) + abs(slippage_amount)))
                    if max_shares <= 0:
                        continue
                    order.quantity = max_shares
                    trade_value = order.quantity * execution_price
                    fees = trade_value * self.commission
            
            # Check if we have enough shares for sell orders
            elif order.quantity < 0:  # Sell order
                current_position = state.get_position(order.symbol)
                max_sell = -current_position
                if order.quantity < max_sell:
                    order.quantity = max_sell
                
                if order.quantity >= 0:  # Can't sell what we don't have
                    continue
            
            # Create fill
            fill = Fill(
                symbol=order.symbol,
                quantity=order.quantity,
                price=execution_price,
                fees=fees,
                slippage=abs(order.quantity) * abs(slippage_amount),
                timestamp=datetime.now()
            )
            
            # Update portfolio state
            if order.quantity > 0:  # Buy
                state.cash -= (trade_value + fees + fill.slippage)
            else:  # Sell
                state.cash += (trade_value - fees - fill.slippage)
            
            state.update_position(order.symbol, order.quantity)
            fills.append(fill)
        
        return fills