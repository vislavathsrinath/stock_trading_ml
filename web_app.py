#!/usr/bin/env python3
"""
Real-time Trading Web Application
Web interface for stock and crypto trading analysis with real-time updates.
"""

from flask import Flask, render_template, jsonify, request
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import json
import os

from data_fetcher import StockDataFetcher
from data_preprocessing import StockDataPreprocessor
from model_training import StockPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_ml_secret_key_2024'

# Global variables for real-time data
data_cache = {}
last_update = {}
update_interval = 300  # 5 minutes

# Assets to track
TRACKED_ASSETS = {
    'stocks': ['TSLA', 'SAP'],
    'crypto': ['BTC-USD', 'BONK-USD'],
    'commodities': ['GC=F', 'SI=F']  # Gold and Silver futures
}

class RealTimeTradingApp:
    """Real-time trading application class."""

    def __init__(self):
        """Initialize the trading app."""
        self.fetcher = StockDataFetcher()
        self.preprocessor = StockDataPreprocessor()
        self.predictors = {}
        self.asset_data = {}

        # Initialize predictors for each asset
        for category, assets in TRACKED_ASSETS.items():
            for asset in assets:
                try:
                    self._initialize_asset_predictor(asset, category)
                except Exception as e:
                    logger.warning(f"Could not initialize predictor for {asset}: {e}")

    def _initialize_asset_predictor(self, symbol, category):
        """Initialize ML predictor for an asset."""
        try:
            # Get historical data for training
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data

            data = self.fetcher.fetch_data(symbol, str(start_date.date()), str(end_date.date()))

            if len(data) < 100:  # Need minimum data
                logger.warning(f"Insufficient data for {symbol}")
                return

            # Preprocess data
            processed_data, X_train, X_val, X_test, y_train, y_val, y_test = \
                self.preprocessor.preprocess_pipeline(data)

            # Train model
            predictor = StockPredictor('random_forest')
            predictor.train(X_train, y_train, X_val, y_val)

            self.predictors[symbol] = predictor
            self.asset_data[symbol] = processed_data

            logger.info(f"Initialized predictor for {symbol}")

        except Exception as e:
            logger.error(f"Error initializing {symbol}: {e}")

    def get_real_time_data(self, symbol):
        """Get real-time data for an asset."""
        try:
            # Check if we need to update
            current_time = time.time()
            if symbol not in last_update or (current_time - last_update[symbol]) > update_interval:
                # Fetch latest data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)  # 60 days for context

                data = self.fetcher.fetch_data(symbol, str(start_date.date()), str(end_date.date()))
                data_cache[symbol] = data
                last_update[symbol] = current_time

                logger.info(f"Updated data for {symbol}")

            return data_cache.get(symbol, pd.DataFrame())

        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {e}")
            return pd.DataFrame()

    def get_asset_analysis(self, symbol):
        """Get comprehensive analysis for an asset."""
        try:
            data = self.get_real_time_data(symbol)
            if data.empty:
                return {}

            # Basic statistics
            latest_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else latest_price
            price_change = latest_price - prev_price
            price_change_pct = (price_change / prev_price) * 100

            # Technical indicators (if available)
            analysis = {
                'symbol': symbol,
                'current_price': round(latest_price, 2),
                'price_change': round(price_change, 2),
                'price_change_pct': round(price_change_pct, 2),
                'volume': int(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(data)
            }

            # Add prediction if model exists
            if symbol in self.predictors:
                try:
                    prediction = self.predictors[symbol].predict_future(self.asset_data.get(symbol, data))
                    analysis['prediction'] = prediction
                except Exception as e:
                    logger.warning(f"Could not get prediction for {symbol}: {e}")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {}

    def get_chart_data(self, symbol, days=30):
        """Get chart data for visualization."""
        try:
            data = self.get_real_time_data(symbol)
            if data.empty:
                return {}

            # Get last N days
            recent_data = data.tail(days)

            chart_data = {
                'dates': [d.strftime('%Y-%m-%d') for d in recent_data['Date']],
                'prices': [round(p, 2) for p in recent_data['Close']],
                'volumes': [int(v) for v in recent_data['Volume']] if 'Volume' in recent_data.columns else []
            }

            return chart_data

        except Exception as e:
            logger.error(f"Error getting chart data for {symbol}: {e}")
            return {}

# Initialize the trading app
trading_app = RealTimeTradingApp()

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html', assets=TRACKED_ASSETS)

@app.route('/api/assets/<symbol>')
def get_asset_data(symbol):
    """API endpoint for asset data."""
    analysis = trading_app.get_asset_analysis(symbol)
    return jsonify(analysis)

@app.route('/api/chart/<symbol>')
def get_chart_data(symbol):
    """API endpoint for chart data."""
    days = int(request.args.get('days', 30))
    chart_data = trading_app.get_chart_data(symbol, days)
    return jsonify(chart_data)

@app.route('/api/all-assets')
def get_all_assets():
    """API endpoint for all tracked assets."""
    all_data = {}
    for category, assets in TRACKED_ASSETS.items():
        all_data[category] = {}
        for asset in assets:
            all_data[category][asset] = trading_app.get_asset_analysis(asset)

    return jsonify(all_data)

@app.route('/asset/<symbol>')
def asset_detail(symbol):
    """Asset detail page."""
    return render_template('asset.html', symbol=symbol)

@app.route('/api/predict/<symbol>')
def predict_asset(symbol):
    """API endpoint for predictions."""
    if symbol in trading_app.predictors:
        try:
            prediction = trading_app.predictors[symbol].predict_future(
                trading_app.asset_data.get(symbol, pd.DataFrame())
            )
            return jsonify(prediction)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No model available for this asset'}), 404

def update_data_background():
    """Background task to update data periodically."""
    while True:
        try:
            logger.info("Updating data in background...")
            for category, assets in TRACKED_ASSETS.items():
                for asset in assets:
                    trading_app.get_real_time_data(asset)
            time.sleep(update_interval)
        except Exception as e:
            logger.error(f"Error in background update: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == '__main__':
    # Start background data update thread
    update_thread = threading.Thread(target=update_data_background, daemon=True)
    update_thread.start()

    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    logger.info("Starting Real-time Trading Web Application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
