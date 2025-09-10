import numpy as np
import pandas as pd
from typing import Dict, Any
from .backtester import BacktestResult


class PerformanceMetrics:
    """Performance metrics calculator for backtest results."""
    
    @staticmethod
    def calculate_metrics(result: BacktestResult, include_benchmark: bool = True) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            result: BacktestResult object
            include_benchmark: Whether to include benchmark comparison metrics
            
        Returns:
            Dictionary of performance metrics
        """
        if not result.equity_curve:
            return {}
        
        df = result.to_dataframe()
        equity = df['Equity']
        
        # Basic metrics
        total_return = (result.final_equity / result.initial_cash) - 1
        days = len(equity)
        years = days / 252.0  # Trading days per year
        
        # Returns
        daily_returns = equity.pct_change().dropna()
        
        # Annualized return
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        annualized_volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_duration = PerformanceMetrics._calculate_drawdown_duration(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = [trade for trade in result.trades if trade.quantity > 0]  # Simplified
        total_trades = len(result.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Best and worst days
        best_day = daily_returns.max() if len(daily_returns) > 0 else 0
        worst_day = daily_returns.min() if len(daily_returns) > 0 else 0
        
        # VaR (Value at Risk) - 5% VaR
        var_5 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        
        metrics = {
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100,
            'Annualized Volatility (%)': annualized_volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown (%)': max_drawdown * 100,
            'Max Drawdown Duration (days)': drawdown_duration,
            'Calmar Ratio': calmar_ratio,
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate * 100,
            'Best Day (%)': best_day * 100,
            'Worst Day (%)': worst_day * 100,
            'VaR 5% (%)': var_5 * 100,
            'Final Equity ($)': result.final_equity,
            'Start Date': result.start_date,
            'End Date': result.end_date,
            'Trading Days': days,
            'Years': years
        }
        
        # Add benchmark comparison metrics if available
        if include_benchmark and result.benchmark_equity_curve:
            benchmark_df = result.get_benchmark_dataframe()
            benchmark_equity = benchmark_df['Equity']
            
            # Benchmark metrics
            benchmark_total_return = (result.benchmark_final_equity / result.initial_cash) - 1
            benchmark_annualized_return = (1 + benchmark_total_return) ** (1/years) - 1 if years > 0 else 0
            benchmark_daily_returns = benchmark_equity.pct_change().dropna()
            benchmark_volatility = benchmark_daily_returns.std() * np.sqrt(252) if len(benchmark_daily_returns) > 1 else 0
            benchmark_sharpe = benchmark_annualized_return / benchmark_volatility if benchmark_volatility > 0 else 0
            
            # Benchmark drawdown
            benchmark_running_max = benchmark_equity.expanding().max()
            benchmark_drawdown = (benchmark_equity - benchmark_running_max) / benchmark_running_max
            benchmark_max_drawdown = benchmark_drawdown.min()
            
            # Comparison metrics
            alpha = (annualized_return - benchmark_annualized_return) * 100
            
            # Beta calculation (if we have enough data)
            beta = 0
            if len(daily_returns) > 1 and len(benchmark_daily_returns) > 1:
                # Align the returns by date
                aligned_returns = pd.DataFrame({
                    'strategy': daily_returns,
                    'benchmark': benchmark_daily_returns
                }).dropna()
                
                if len(aligned_returns) > 1:
                    cov_matrix = np.cov(aligned_returns['strategy'], aligned_returns['benchmark'])
                    if len(cov_matrix) == 2 and len(cov_matrix[0]) == 2:
                        beta = cov_matrix[0, 1] / np.var(aligned_returns['benchmark']) if np.var(aligned_returns['benchmark']) != 0 else 0
            
            # Information ratio
            tracking_error = 0
            information_ratio = 0
            if len(daily_returns) > 1 and len(benchmark_daily_returns) > 1:
                aligned_returns = pd.DataFrame({
                    'strategy': daily_returns,
                    'benchmark': benchmark_daily_returns
                }).dropna()
                
                if len(aligned_returns) > 1:
                    excess_returns = aligned_returns['strategy'] - aligned_returns['benchmark']
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    information_ratio = alpha / 100 / tracking_error if tracking_error != 0 else 0
            
            # Add benchmark metrics
            metrics.update({
                'Benchmark Total Return (%)': benchmark_total_return * 100,
                'Benchmark Annualized Return (%)': benchmark_annualized_return * 100,
                'Benchmark Volatility (%)': benchmark_volatility * 100,
                'Benchmark Sharpe Ratio': benchmark_sharpe,
                'Benchmark Max Drawdown (%)': benchmark_max_drawdown * 100,
                'Alpha (%)': alpha,
                'Beta': beta,
                'Tracking Error (%)': tracking_error * 100,
                'Information Ratio': information_ratio
            })
        
        return metrics
    
    @staticmethod
    def _calculate_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        if drawdown.empty:
            return 0
        
        in_drawdown = drawdown < 0
        
        if not in_drawdown.any():
            return 0
        
        # Find drawdown periods
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
        
        if not drawdown_starts.any() or not drawdown_ends.any():
            return len(drawdown) if in_drawdown.iloc[-1] else 0
        
        starts = drawdown.index[drawdown_starts]
        ends = drawdown.index[drawdown_ends]
        
        # Handle case where we end in drawdown
        if len(starts) > len(ends):
            ends = ends.append(pd.Index([drawdown.index[-1]]))
        
        # Calculate durations
        durations = [(end - start).days for start, end in zip(starts[:len(ends)], ends)]
        
        return max(durations) if durations else 0
    
    @staticmethod
    def print_metrics(metrics: Dict[str, Any]) -> None:
        """Print metrics in a formatted way."""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        
        # Format and print each metric
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key.endswith('(%)'):
                    print(f"{key:<30}: {value:>10.2f}%")
                elif key.endswith('($)'):
                    print(f"{key:<30}: ${value:>13,.2f}")
                elif key in ['Sharpe Ratio', 'Calmar Ratio']:
                    print(f"{key:<30}: {value:>10.2f}")
                elif key in ['Trading Days', 'Total Trades']:
                    print(f"{key:<30}: {value:>10.0f}")
                elif key == 'Years':
                    print(f"{key:<30}: {value:>10.1f}")
                else:
                    print(f"{key:<30}: {value:>10.2f}")
            else:
                print(f"{key:<30}: {value}")
        
        print("="*50)