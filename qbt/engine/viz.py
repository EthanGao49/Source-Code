import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .backtester import BacktestResult


class Visualizer:
    """Visualization utilities for backtest results."""
    
    @staticmethod
    def plot_equity_curve(
        result: BacktestResult,
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot the equity curve over time.
        
        Args:
            result: BacktestResult object
            benchmark: Optional benchmark series for comparison
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        df = result.to_dataframe()
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Plot equity curve
        ax.plot(df.index, df['Equity'], label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values, label='Benchmark', 
                   linewidth=2, color='gray', alpha=0.7)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_drawdown(
        result: BacktestResult,
        title: str = "Portfolio Drawdown",
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """
        Plot the drawdown over time.
        
        Args:
            result: BacktestResult object
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        df = result.to_dataframe()
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate drawdown
        equity = df['Equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        # Plot drawdown
        ax.fill_between(df.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(df.index, drawdown, color='red', linewidth=1)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_returns_distribution(
        result: BacktestResult,
        title: str = "Daily Returns Distribution",
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot the distribution of daily returns.
        
        Args:
            result: BacktestResult object
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        df = result.to_dataframe()
        
        if df.empty:
            ax1.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Calculate daily returns
        daily_returns = df['Equity'].pct_change().dropna() * 100
        
        # Histogram
        ax1.hist(daily_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(daily_returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {daily_returns.mean():.2f}%')
        ax1.set_title('Returns Histogram', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Daily Returns (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot (simplified)
        from scipy import stats
        stats.probplot(daily_returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot vs Normal', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_monthly_returns_heatmap(
        result: BacktestResult,
        title: str = "Monthly Returns Heatmap",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot monthly returns as a heatmap.
        
        Args:
            result: BacktestResult object
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        df = result.to_dataframe()
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate monthly returns
        monthly_equity = df['Equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100
        
        if len(monthly_returns) == 0:
            ax.text(0.5, 0.5, 'Insufficient data for monthly analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot_data = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).mean().unstack(fill_value=0)
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', fontsize=12)
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_comprehensive_analysis(
        result: BacktestResult,
        benchmark: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive analysis plot with multiple subplots.
        
        Args:
            result: BacktestResult object
            benchmark: Optional benchmark series
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        df = result.to_dataframe()
        
        if not df.empty:
            ax1.plot(df.index, df['Equity'], label='Portfolio', linewidth=2, color='blue')
            if benchmark is not None:
                ax1.plot(benchmark.index, benchmark.values, label='Benchmark', 
                        linewidth=2, color='gray', alpha=0.7)
            ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        if not df.empty:
            equity = df['Equity']
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100
            ax2.fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
            ax2.plot(df.index, drawdown, color='red', linewidth=1)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        ax3 = fig.add_subplot(gs[2, 0])
        if not df.empty:
            daily_returns = df['Equity'].pct_change().dropna() * 100
            ax3.hist(daily_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(daily_returns.mean(), color='red', linestyle='--')
            ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Daily Returns (%)', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # Rolling metrics
        ax4 = fig.add_subplot(gs[2, 1])
        if not df.empty and len(df) > 20:
            daily_returns = df['Equity'].pct_change().dropna()
            rolling_sharpe = (daily_returns.rolling(20).mean() / daily_returns.rolling(20).std() * np.sqrt(252))
            ax4.plot(rolling_sharpe.index, rolling_sharpe, color='green', linewidth=2)
            ax4.set_title('Rolling 20-Day Sharpe Ratio', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Sharpe Ratio', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        # Format dates for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            if hasattr(ax, 'xaxis'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Comprehensive Backtest Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig