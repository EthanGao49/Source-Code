import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import io
import base64
from .backtester import BacktestResult
from .metrics import PerformanceMetrics
from .viz import Visualizer


class SummaryReport:
    """Generate comprehensive PDF reports for backtest results."""
    
    def __init__(self, result: BacktestResult):
        """
        Initialize summary report.
        
        Args:
            result: BacktestResult object (config is read from result.config)
        """
        self.result = result
        self.config = result.config
        self.metrics = PerformanceMetrics.calculate_metrics(result)
        
    def generate_pdf(self, filename: str = None) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            filename: Output filename. If None, auto-generate based on timestamp
            
        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.pdf"
        
        with PdfPages(filename) as pdf:
            # Title page
            self._create_title_page(pdf)
            
            # Configuration summary
            self._create_config_page(pdf)
            
            # Performance metrics
            self._create_metrics_page(pdf)
            
            # Benchmark comparison (if available)
            if self.result.benchmark_equity_curve:
                self._create_benchmark_comparison_page(pdf)
            
            # Equity curve and visualizations
            self._create_equity_plots_page(pdf)
            
            # Drawdown analysis
            self._create_drawdown_page(pdf)
            
            # Returns analysis
            self._create_returns_page(pdf)
            
            # Order history
            self._create_order_history_page(pdf)
            
            # Trade analysis
            self._create_trade_analysis_page(pdf)
            
            # Monthly performance heatmap
            try:
                self._create_monthly_heatmap_page(pdf)
            except Exception as e:
                print(f"Could not generate monthly heatmap: {e}")
        
        print(f"PDF report generated: {filename}")
        return filename
    
    def _create_title_page(self, pdf: PdfPages):
        """Create title page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, 'Quantitative Backtesting Report', 
               ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Subtitle
        strategy_name = self.config.get('strategy_name', 'Trading Strategy')
        ax.text(0.5, 0.75, strategy_name, 
               ha='center', va='center', fontsize=16, style='italic')
        
        # Summary stats
        summary_text = f"""
        Universe: {', '.join(self.config.get('universe', []))}
        Period: {self.result.start_date.strftime('%Y-%m-%d')} to {self.result.end_date.strftime('%Y-%m-%d')}
        Initial Capital: ${self.result.initial_cash:,.2f}
        Final Equity: ${self.result.final_equity:,.2f}
        Total Return: {((self.result.final_equity / self.result.initial_cash - 1) * 100):.2f}%
        """
        
        ax.text(0.5, 0.55, summary_text, ha='center', va='center', 
               fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        # Generation info
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.2, f'Report generated: {generation_time}', 
               ha='center', va='center', fontsize=10, style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_config_page(self, pdf: PdfPages):
        """Create configuration summary page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Backtest Configuration', 
               ha='center', va='top', fontsize=18, fontweight='bold')
        
        # Create configuration text
        config_lines = []
        
        # Basic parameters
        config_lines.append("BASIC PARAMETERS")
        config_lines.append("=" * 50)
        config_lines.append(f"Universe: {', '.join(self.config.get('universe', []))}")
        config_lines.append(f"Start Date: {self.result.start_date}")
        config_lines.append(f"End Date: {self.result.end_date}")
        config_lines.append(f"Initial Cash: ${self.result.initial_cash:,.2f}")
        config_lines.append(f"Trading Interval: {self.config.get('interval', '1d')}")
        config_lines.append("")
        
        # Strategy configuration
        config_lines.append("STRATEGY CONFIGURATION")
        config_lines.append("=" * 50)
        strategy_config = self.config.get('strategy', {})
        for key, value in strategy_config.items():
            config_lines.append(f"{key}: {value}")
        config_lines.append("")
        
        # Signal generators
        config_lines.append("SIGNAL GENERATORS")
        config_lines.append("=" * 50)
        signals_config = self.config.get('signals', [])
        for i, signal in enumerate(signals_config, 1):
            config_lines.append(f"{i}. {signal}")
        config_lines.append("")
        
        # Broker configuration
        config_lines.append("BROKER CONFIGURATION")
        config_lines.append("=" * 50)
        broker_config = self.config.get('broker', {})
        for key, value in broker_config.items():
            config_lines.append(f"{key}: {value}")
        
        # Benchmark configuration
        if self.result.benchmark_equity_curve:
            config_lines.append("")
            config_lines.append("BENCHMARK CONFIGURATION")
            config_lines.append("=" * 50)
            benchmark_config = self.config.get('benchmark', {})
            for key, value in benchmark_config.items():
                config_lines.append(f"{key}: {value}")
        
        config_text = '\n'.join(config_lines)
        ax.text(0.05, 0.9, config_text, ha='left', va='top', fontsize=10, 
               fontfamily='monospace', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_metrics_page(self, pdf: PdfPages):
        """Create performance metrics page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Performance Metrics', 
               ha='center', va='top', fontsize=18, fontweight='bold')
        
        # Format metrics into two columns
        metrics_lines = []
        
        # Core metrics
        metrics_lines.append("RETURN METRICS")
        metrics_lines.append("=" * 40)
        metrics_lines.append(f"Total Return: {self.metrics.get('Total Return (%)', 0):.2f}%")
        metrics_lines.append(f"Annualized Return: {self.metrics.get('Annualized Return (%)', 0):.2f}%")
        metrics_lines.append(f"Final Equity: ${self.metrics.get('Final Equity ($)', 0):,.2f}")
        metrics_lines.append("")
        
        # Risk metrics
        metrics_lines.append("RISK METRICS")
        metrics_lines.append("=" * 40)
        metrics_lines.append(f"Annualized Volatility: {self.metrics.get('Annualized Volatility (%)', 0):.2f}%")
        metrics_lines.append(f"Maximum Drawdown: {self.metrics.get('Maximum Drawdown (%)', 0):.2f}%")
        metrics_lines.append(f"Max Drawdown Duration: {self.metrics.get('Max Drawdown Duration (days)', 0):.0f} days")
        metrics_lines.append(f"VaR (5%): {self.metrics.get('VaR 5% (%)', 0):.2f}%")
        metrics_lines.append("")
        
        # Risk-adjusted metrics
        metrics_lines.append("RISK-ADJUSTED METRICS")
        metrics_lines.append("=" * 40)
        metrics_lines.append(f"Sharpe Ratio: {self.metrics.get('Sharpe Ratio', 0):.2f}")
        metrics_lines.append(f"Calmar Ratio: {self.metrics.get('Calmar Ratio', 0):.2f}")
        metrics_lines.append("")
        
        # Trading metrics
        metrics_lines.append("TRADING METRICS")
        metrics_lines.append("=" * 40)
        metrics_lines.append(f"Total Trades: {self.metrics.get('Total Trades', 0):.0f}")
        metrics_lines.append(f"Win Rate: {self.metrics.get('Win Rate (%)', 0):.2f}%")
        metrics_lines.append(f"Best Day: {self.metrics.get('Best Day (%)', 0):.2f}%")
        metrics_lines.append(f"Worst Day: {self.metrics.get('Worst Day (%)', 0):.2f}%")
        metrics_lines.append("")
        
        # Time metrics
        metrics_lines.append("TIME METRICS")
        metrics_lines.append("=" * 40)
        metrics_lines.append(f"Trading Days: {self.metrics.get('Trading Days', 0):.0f}")
        metrics_lines.append(f"Years: {self.metrics.get('Years', 0):.1f}")
        
        metrics_text = '\n'.join(metrics_lines)
        ax.text(0.05, 0.9, metrics_text, ha='left', va='top', fontsize=11, 
               fontfamily='monospace', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_benchmark_comparison_page(self, pdf: PdfPages):
        """Create benchmark comparison page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Benchmark Comparison', 
               ha='center', va='top', fontsize=18, fontweight='bold')
        
        # Benchmark comparison metrics
        comparison_lines = []
        
        comparison_lines.append("STRATEGY vs BENCHMARK")
        comparison_lines.append("=" * 50)
        comparison_lines.append(f"Strategy Return: {self.metrics.get('Total Return (%)', 0):.2f}%")
        comparison_lines.append(f"Benchmark Return: {self.metrics.get('Benchmark Total Return (%)', 0):.2f}%")
        comparison_lines.append(f"Alpha: {self.metrics.get('Alpha (%)', 0):.2f}%")
        comparison_lines.append("")
        
        comparison_lines.append("ANNUALIZED COMPARISON")
        comparison_lines.append("=" * 50)
        comparison_lines.append(f"Strategy Ann. Return: {self.metrics.get('Annualized Return (%)', 0):.2f}%")
        comparison_lines.append(f"Benchmark Ann. Return: {self.metrics.get('Benchmark Annualized Return (%)', 0):.2f}%")
        comparison_lines.append("")
        
        comparison_lines.append("RISK COMPARISON")
        comparison_lines.append("=" * 50)
        comparison_lines.append(f"Strategy Volatility: {self.metrics.get('Annualized Volatility (%)', 0):.2f}%")
        comparison_lines.append(f"Benchmark Volatility: {self.metrics.get('Benchmark Volatility (%)', 0):.2f}%")
        comparison_lines.append(f"Strategy Max DD: {self.metrics.get('Maximum Drawdown (%)', 0):.2f}%")
        comparison_lines.append(f"Benchmark Max DD: {self.metrics.get('Benchmark Max Drawdown (%)', 0):.2f}%")
        comparison_lines.append("")
        
        comparison_lines.append("RISK-ADJUSTED COMPARISON")
        comparison_lines.append("=" * 50)
        comparison_lines.append(f"Strategy Sharpe: {self.metrics.get('Sharpe Ratio', 0):.2f}")
        comparison_lines.append(f"Benchmark Sharpe: {self.metrics.get('Benchmark Sharpe Ratio', 0):.2f}")
        comparison_lines.append("")
        
        comparison_lines.append("RELATIVE PERFORMANCE")
        comparison_lines.append("=" * 50)
        comparison_lines.append(f"Beta: {self.metrics.get('Beta', 0):.2f}")
        comparison_lines.append(f"Tracking Error: {self.metrics.get('Tracking Error (%)', 0):.2f}%")
        comparison_lines.append(f"Information Ratio: {self.metrics.get('Information Ratio', 0):.2f}")
        
        comparison_text = '\n'.join(comparison_lines)
        ax.text(0.05, 0.9, comparison_text, ha='left', va='top', fontsize=11, 
               fontfamily='monospace', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_equity_plots_page(self, pdf: PdfPages):
        """Create equity curve plots page."""
        # Equity curve
        fig = Visualizer.plot_equity_curve(self.result, title="Portfolio Equity Curve")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Comprehensive analysis
        try:
            fig = Visualizer.plot_comprehensive_analysis(self.result)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Could not generate comprehensive analysis plot: {e}")
    
    def _create_drawdown_page(self, pdf: PdfPages):
        """Create drawdown analysis page."""
        fig = Visualizer.plot_drawdown(self.result)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_returns_page(self, pdf: PdfPages):
        """Create returns distribution page."""
        fig = Visualizer.plot_returns_distribution(self.result)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_monthly_heatmap_page(self, pdf: PdfPages):
        """Create monthly returns heatmap page."""
        fig = Visualizer.plot_monthly_returns_heatmap(self.result)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_order_history_page(self, pdf: PdfPages):
        """Create order history page."""
        trades_df = self.result.get_trades_dataframe()
        
        if trades_df.empty:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.5, 'No trades executed', ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Create multiple pages if needed for large trade history
        trades_per_page = 50
        num_pages = (len(trades_df) - 1) // trades_per_page + 1
        
        for page in range(num_pages):
            start_idx = page * trades_per_page
            end_idx = min((page + 1) * trades_per_page, len(trades_df))
            page_trades = trades_df.iloc[start_idx:end_idx]
            
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            title = f'Order History (Page {page + 1} of {num_pages})'
            ax.text(0.5, 0.95, title, ha='center', va='top', 
                   fontsize=16, fontweight='bold')
            
            # Format trades table
            table_data = []
            table_data.append(['Date', 'Symbol', 'Quantity', 'Price', 'Fees', 'Total Cost'])
            
            for idx, (date, trade) in enumerate(page_trades.iterrows()):
                table_data.append([
                    date.strftime('%Y-%m-%d'),
                    trade['Symbol'],
                    f"{trade['Quantity']:.0f}",
                    f"${trade['Price']:.2f}",
                    f"${trade['Fees']:.2f}",
                    f"${trade['Total_Cost']:.2f}"
                ])
            
            # Create table
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='center', loc='center', bbox=[0.05, 0.1, 0.9, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            # Style header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def _create_trade_analysis_page(self, pdf: PdfPages):
        """Create trade analysis page."""
        trades_df = self.result.get_trades_dataframe()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        if trades_df.empty:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No trades to analyze', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Trade size distribution
        ax1.hist(trades_df['Quantity'], bins=20, alpha=0.7, color='blue')
        ax1.set_title('Trade Size Distribution')
        ax1.set_xlabel('Quantity')
        ax1.set_ylabel('Frequency')
        
        # Trade cost distribution
        ax2.hist(trades_df['Total_Cost'], bins=20, alpha=0.7, color='green')
        ax2.set_title('Trade Cost Distribution')
        ax2.set_xlabel('Total Cost ($)')
        ax2.set_ylabel('Frequency')
        
        # Trades by symbol
        symbol_counts = trades_df['Symbol'].value_counts()
        if len(symbol_counts) <= 10:
            ax3.bar(symbol_counts.index, symbol_counts.values, color='orange')
            ax3.set_title('Trades by Symbol')
            ax3.set_xlabel('Symbol')
            ax3.set_ylabel('Number of Trades')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.pie(symbol_counts.head(10).values, labels=symbol_counts.head(10).index, autopct='%1.1f%%')
            ax3.set_title('Top 10 Symbols by Trade Count')
        
        # Trades over time
        monthly_trades = trades_df.resample('M').size()
        ax4.plot(monthly_trades.index, monthly_trades.values, marker='o', color='red')
        ax4.set_title('Trades Over Time')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Number of Trades')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


