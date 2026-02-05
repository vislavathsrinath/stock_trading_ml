"""
Data Fetcher Module
Handles fetching financial data from various sources.
"""

import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Fetches stock and financial data from Yahoo Finance."""

    def __init__(self):
        """Initialize the data fetcher."""
        self.valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical data for a symbol.

        Args:
            symbol (str): Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d', '1h', '1m', etc.)

        Returns:
            pd.DataFrame: Historical data with OHLCV columns
        """
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

            # Validate interval
            if interval not in self.valid_intervals:
                logger.warning(f"Invalid interval {interval}, using '1d'")
                interval = '1d'

            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Reset index to make Date a column
            data = data.reset_index()

            # Ensure Date column is datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            elif 'Datetime' in data.columns:
                data['Date'] = pd.to_datetime(data['Datetime'])
                data = data.drop('Datetime', axis=1)

            # Rename columns to standard format
            column_mapping = {
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'Adj Close': 'Adj Close'
            }

            data = data.rename(columns=column_mapping)

            # Ensure required columns exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                logger.warning(f"Missing columns for {symbol}: {missing_columns}")
                # Add missing columns with NaN values
                for col in missing_columns:
                    data[col] = pd.NA

            # Sort by date
            data = data.sort_values('Date').reset_index(drop=True)

            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.

        Args:
            symbol (str): Stock/crypto symbol

        Returns:
            float or None: Latest closing price
        """
        try:
            # Get last 5 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)

            data = self.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None

    def get_company_info(self, symbol: str) -> dict:
        """
        Get company information.

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None)
            }

        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get a small amount of data
            data = ticker.history(period='1d')
            return not data.empty
        except Exception:
            return False

    def get_multiple_symbols(self, symbols: list, start_date: str, end_date: str,
                           interval: str = '1d') -> dict:
        """
        Fetch data for multiple symbols.

        Args:
            symbols (list): List of symbols
            start_date (str): Start date
            end_date (str): End date
            interval (str): Data interval

        Returns:
            dict: Dictionary of symbol -> DataFrame
        """
        results = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval)
            if not data.empty:
                results[symbol] = data

        return results
