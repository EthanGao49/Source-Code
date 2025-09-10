import numpy as np
import pandas as pd
from typing import Dict, Any
from .backtester import BacktestResult


class PerformanceMetrics:
    """Performance metrics calculator for backtest results."""
    
    @staticmethod
    def calculate_metrics(result: BacktestResult, include_benchmark: bool = True, benchmark_name: str = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            result: BacktestResult object
            include_benchmark: Whether to include benchmark comparison metrics
            benchmark_name: Specific benchmark name to compare against (for multiple benchmarks)
            
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
        benchmark_data = None
        benchmark_final_equity = 0
        
        # Handle multiple benchmarks
        if include_benchmark and benchmark_name and benchmark_name in result.benchmarks:
            benchmark_data = result.benchmarks[benchmark_name]
            benchmark_final_equity = benchmark_data['final_equity']
            benchmark_df = result.get_benchmark_dataframe(benchmark_name)
        # Auto-select first benchmark if none specified but benchmarks exist
        elif include_benchmark and not benchmark_name and result.benchmarks:
            # Use the first available benchmark
            first_benchmark_name = list(result.benchmarks.keys())[0]
            benchmark_data = result.benchmarks[first_benchmark_name]
            benchmark_final_equity = benchmark_data['final_equity']
            benchmark_df = result.get_benchmark_dataframe(first_benchmark_name)
            benchmark_name = first_benchmark_name  # Set for later use in suffix
        # Legacy single benchmark support
        elif include_benchmark and result.benchmark_equity_curve:
            benchmark_final_equity = result.benchmark_final_equity
            benchmark_df = result.get_benchmark_dataframe()
        
        if include_benchmark and (
            (benchmark_name and benchmark_name in result.benchmarks) or 
            (not benchmark_name and result.benchmark_equity_curve) or
            (benchmark_data is not None)  # For auto-selected benchmark
        ):
            if not benchmark_df.empty:
                benchmark_equity = benchmark_df['Equity']
                
                # Benchmark metrics
                benchmark_total_return = (benchmark_final_equity / result.initial_cash) - 1
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
                
                # Add benchmark metrics with name suffix if specified
                suffix = f" ({benchmark_name})" if benchmark_name else ""
                metrics.update({
                    f'Benchmark Total Return (%)': benchmark_total_return * 100,
                    f'Benchmark Annualized Return (%)': benchmark_annualized_return * 100,
                    f'Benchmark Volatility (%)': benchmark_volatility * 100,
                    f'Benchmark Sharpe Ratio': benchmark_sharpe,
                    f'Benchmark Max Drawdown (%)': benchmark_max_drawdown * 100,
                    f'Alpha (%)': alpha,
                    f'Beta': beta,
                    f'Tracking Error (%)': tracking_error * 100,
                    f'Information Ratio': information_ratio
                })
        
        return metrics
    
    @staticmethod
    def calculate_benchmark_standalone_metrics(result: BacktestResult, benchmark_name: str) -> Dict[str, Any]:
        """
        Calculate standalone metrics for a specific benchmark.
        
        Args:
            result: BacktestResult object
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary of benchmark metrics
        """
        if benchmark_name not in result.benchmarks:
            return {}
        
        benchmark_data = result.benchmarks[benchmark_name]
        benchmark_df = result.get_benchmark_dataframe(benchmark_name)
        
        if benchmark_df.empty:
            return {}
        
        equity = benchmark_df['Equity']
        
        # Basic metrics
        benchmark_final_equity = benchmark_data['final_equity']
        total_return = (benchmark_final_equity / result.initial_cash) - 1
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
        
        # Best and worst days
        best_day = daily_returns.max() if len(daily_returns) > 0 else 0
        worst_day = daily_returns.min() if len(daily_returns) > 0 else 0
        
        # VaR (Value at Risk) - 5% VaR
        var_5 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        
        return {
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100,
            'Annualized Volatility (%)': annualized_volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown (%)': max_drawdown * 100,
            'Max Drawdown Duration (days)': drawdown_duration,
            'Calmar Ratio': calmar_ratio,
            'Best Day (%)': best_day * 100,
            'Worst Day (%)': worst_day * 100,
            'VaR 5% (%)': var_5 * 100,
            'Final Equity ($)': benchmark_final_equity,
            'Start Date': result.start_date,
            'End Date': result.end_date,
            'Trading Days': days,
            'Years': years
        }

    @staticmethod
    def calculate_all_benchmark_metrics(result: BacktestResult) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics for strategy and all benchmarks with comparisons.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary mapping names to their metrics (includes strategy vs benchmark comparisons)
        """
        all_metrics = {}
        
        # Calculate strategy vs each benchmark comparison
        for benchmark_name in result.get_benchmark_names():
            comparison_metrics = PerformanceMetrics.calculate_metrics(
                result, include_benchmark=True, benchmark_name=benchmark_name
            )
            all_metrics[f'Strategy vs {benchmark_name}'] = comparison_metrics
        
        # Calculate standalone benchmark metrics
        for benchmark_name in result.get_benchmark_names():
            standalone_metrics = PerformanceMetrics.calculate_benchmark_standalone_metrics(
                result, benchmark_name
            )
            if standalone_metrics:
                all_metrics[f'{benchmark_name} Standalone'] = standalone_metrics
        
        return all_metrics
    
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