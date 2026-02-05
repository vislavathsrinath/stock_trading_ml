#!/usr/bin/env python3
"""
Real-Time Data Fetcher Module
Handles fetching real-time financial data from various APIs.
"""

import pandas as pd
import yfinance as yf
import requests
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import os

logger = logging.getLogger(__name__)

class RealTimeDataFetcher:
    """Fetches real-time financial data from multiple APIs."""

    def __init__(self):
        """Initialize the real-time data fetcher."""
        self.valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.alpha_vantage_base_url = "https://www.alphavantage.co/api/v3"
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

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

            # Try different data sources based on symbol type
            if symbol.endswith('-USD') or symbol in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']:
                # Crypto symbol - try CoinGecko first
                data = self._fetch_crypto_historical(symbol.replace('-USD', '').lower(), start_date, end_date)
                if not data.empty:
                    return data

            # Fallback to Yahoo Finance
            return self._fetch_yahoo_data(symbol, start_date, end_date, interval)

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_real_time_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time price for a symbol.

        Args:
            symbol (str): Asset symbol

        Returns:
            dict: Real-time price data or None
        """
        try:
            if symbol.endswith('-USD') or symbol in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']:
                # Crypto - use CoinGecko
                return self._get_crypto_price(symbol.replace('-USD', '').lower())
            elif symbol in ['TSLA', 'AAPL', 'GOOGL', 'MSFT', 'SAP']:
                # Stock - use Alpha Vantage if API key available
                if self.alpha_vantage_api_key:
                    return self._get_stock_price_alpha_vantage(symbol)
                else:
                    # Fallback to Yahoo Finance latest price
                    return self._get_yahoo_latest_price(symbol)
            else:
                # Other assets - use Yahoo Finance
                return self._get_yahoo_latest_price(symbol)

        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {str(e)}")
            return None

    def _fetch_crypto_historical(self, crypto_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical crypto data from CoinGecko."""
        try:
            # Convert dates to timestamps
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())

            url = f"{self.coingecko_base_url}/coins/{crypto_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_ts,
                'to': end_ts
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'prices' not in data or not data['prices']:
                return pd.DataFrame()

            # Create DataFrame from prices data
            prices = data['prices']
            market_caps = data.get('market_caps', [])
            volumes = data.get('total_volumes', [])

            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                date = datetime.fromtimestamp(timestamp / 1000)

                # Get corresponding market cap and volume
                market_cap = market_caps[i][1] if i < len(market_caps) else 0
                volume = volumes[i][1] if i < len(volumes) else 0

                df_data.append({
                    'Date': date,
                    'Open': price,  # Approximation
                    'High': price,  # Approximation
                    'Low': price,   # Approximation
                    'Close': price,
                    'Volume': volume,
                    'Market_Cap': market_cap
                })

            df = pd.DataFrame(df_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').reset_index()

            return df

        except Exception as e:
            logger.error(f"Error fetching crypto historical data: {str(e)}")
            return pd.DataFrame()

    def _get_crypto_price(self, crypto_id: str) -> Optional[Dict[str, Any]]:
        """Get real-time crypto price from CoinGecko."""
        try:
            url = f"{self.coingecko_base_url}/simple/price"
            params = {
                'ids': crypto_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if crypto_id not in data:
                return None

            price_data = data[crypto_id]
            current_price = price_data.get('usd', 0)
            change_24h = price_data.get('usd_24h_change', 0)
            volume_24h = price_data.get('usd_24h_vol', 0)

            return {
                'symbol': f"{crypto_id.upper()}-USD",
                'price': current_price,
                'change': (change_24h / 100) * current_price,  # Convert percentage to absolute
                'change_percent': change_24h,
                'volume': volume_24h,
                'timestamp': datetime.now(),
                'source': 'CoinGecko'
            }

        except Exception as e:
            logger.error(f"Error fetching crypto price: {str(e)}")
            return None

    def _get_stock_price_alpha_vantage(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time stock price from Alpha Vantage."""
        try:
            if not self.alpha_vantage_api_key:
                return None

            url = f"{self.alpha_vantage_base_url}/quote"
            params = {
                'symbol': symbol,
                'apikey': self.alpha_vantage_api_key
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'Global Quote' not in data:
                return None

            quote = data['Global Quote']

            return {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': float(quote.get('10. change percent', '0').strip('%')),
                'volume': int(quote.get('06. volume', 0)),
                'timestamp': datetime.now(),
                'source': 'Alpha Vantage'
            }

        except Exception as e:
            logger.error(f"Error fetching stock price from Alpha Vantage: {str(e)}")
            return None

    def _get_yahoo_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')

            if data.empty:
                return None

            latest = data.iloc[-1]
            prev_close = data.iloc[-2]['Close'] if len(data) > 1 else latest['Close']

            change = latest['Close'] - prev_close
            change_percent = (change / prev_close) * 100

            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'change': float(change),
                'change_percent': float(change_percent),
                'volume': int(latest['Volume']) if 'Volume' in latest.index else 0,
                'timestamp': datetime.now(),
                'source': 'Yahoo Finance'
            }

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance price: {str(e)}")
            return None

    def _fetch_yahoo_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
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

            return data

        except Exception as e:
            logger.error(f"Error fetching Yahoo data: {str(e)}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol (backward compatibility).

        Args:
            symbol (str): Asset symbol

        Returns:
            float: Latest price or None
        """
        real_time_data = self.get_real_time_price(symbol)
        if real_time_data:
            return real_time_data['price']

        # Fallback to historical data
        try:
            if symbol.endswith('-USD'):
                # Try crypto
                crypto_data = self._get_crypto_price(symbol.replace('-USD', '').lower())
                if crypto_data:
                    return crypto_data['price']

            # Try Yahoo Finance
            data = self._fetch_yahoo_data(symbol, '2024-01-01', datetime.now().strftime('%Y-%m-%d'))
            if not data.empty:
                return float(data['Close'].iloc[-1])
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
            # Try real-time price first
            if self.get_real_time_price(symbol):
                return True

            # Fallback to Yahoo Finance
            ticker = yf.Ticker(symbol)
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
