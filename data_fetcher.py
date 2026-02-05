#!/usr/bin/env python3
"""
Stock Data Fetcher Module
Handles fetching historical stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Class for fetching stock data."""

    def __init__(self):
        """Initialize the data fetcher."""
        self.ticker = None

    def fetch_data(self, symbol, start_date, end_date, interval='1d'):
        """
        Fetch historical stock data.

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('1d', '1h', '1m', etc.)

        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

            # Create ticker object
            self.ticker = yf.Ticker(symbol)

            # Fetch historical data
            data = self.ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Reset index to make Date a column
            data = data.reset_index()

            # Convert Date to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                data['Date'] = pd.to_datetime(data['Date'])

            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def get_company_info(self, symbol):
        """
        Get basic company information.

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Company information
        """
        try:
            if self.ticker is None or self.ticker.ticker != symbol:
                self.ticker = yf.Ticker(symbol)

            info = self.ticker.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }

        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {str(e)}")
            return {}

    def get_multiple_stocks(self, symbols, start_date, end_date, interval='1d'):
        """
        Fetch data for multiple stocks.

        Args:
            symbols (list): List of stock symbols
            start_date (str): Start date
            end_date (str): End date
            interval (str): Data interval

        Returns:
            dict: Dictionary of DataFrames keyed by symbol
        """
        data_dict = {}
        for symbol in symbols:
            try:
                data_dict[symbol] = self.fetch_data(symbol, start_date, end_date, interval)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                continue

        return data_dict
