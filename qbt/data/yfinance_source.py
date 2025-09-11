from typing import List
from datetime import datetime
import pandas as pd
from .base import DataSource


class YFDataSource(DataSource):
    """Yahoo Finance data source implementation."""
    
    def get_price(
        self,
        universe: List[str],
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        import yfinance as yf

        if not universe:
            return pd.DataFrame()

        result_data = []

        data = yf.download(
            tickers=universe,
            start=start,
            end=end,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            prepost=True,
            threads=True
        )
        if data.empty:
            return pd.DataFrame()
        for symbol in universe:
            try:
                symbol_data = data[symbol].copy()
                if symbol_data.empty or symbol_data.isna().all().all():
                    continue
                symbol_data = symbol_data.dropna()
                if not symbol_data.empty:
                    symbol_data['Symbol'] = symbol
                    symbol_data = symbol_data.reset_index()
                    symbol_data = symbol_data.set_index(['Date', 'Symbol'])
                    result_data.append(symbol_data)
            except (KeyError, AttributeError):
                continue

        if not result_data:
            return pd.DataFrame()

        result_df = pd.concat(result_data)
        result_df.index.names = ['Date', 'Symbol']

        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = set(expected_columns) - set(result_df.columns)
        for col in missing_columns:
            if col == 'Volume':
                result_df[col] = 0
            else:
                result_df[col] = result_df['Close'] if 'Close' in result_df.columns else 0

        result_df = result_df[expected_columns]
        return result_df.sort_index()