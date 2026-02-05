#!/usr/bin/env python3
"""
Main Stock Trading ML Application
Orchestrates the entire pipeline from data fetching to model training and prediction.
"""

import logging
import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

from data_fetcher import StockDataFetcher
from data_preprocessing import StockDataPreprocessor
from model_training import StockPredictor, ModelComparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_trading_ml.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockTradingML:
    """Main application class for stock trading ML."""

    def __init__(self):
        """Initialize the application."""
        self.fetcher = StockDataFetcher()
        self.preprocessor = StockDataPreprocessor()
        self.predictor = None
        self.data = None
        self.processed_data = None

    def run_pipeline(self, symbol, start_date, end_date, model_type='random_forest',
                    prediction_days=1, test_size=0.2):
        """
        Run the complete ML pipeline.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
            model_type (str): ML model type
            prediction_days (int): Days ahead to predict
            test_size (float): Test set size
        """
        try:
            logger.info("Starting stock trading ML pipeline...")

            # 1. Fetch data
            logger.info("Step 1: Fetching data...")
            self.data = self.fetcher.fetch_data(symbol, start_date, end_date)

            # 2. Preprocess data
            logger.info("Step 2: Preprocessing data...")
            self.processed_data, X_train, X_val, X_test, y_train, y_val, y_test = \
                self.preprocessor.preprocess_pipeline(self.data, prediction_days)

            # 3. Train model
            logger.info("Step 3: Training model...")
            self.predictor = StockPredictor(model_type)
            self.predictor.train(X_train, y_train, X_val, y_val)

            # 4. Evaluate model
            logger.info("Step 4: Evaluating model...")
            test_metrics = self.predictor.evaluate(X_test, y_test)

            # 5. Save model
            logger.info("Step 5: Saving model...")
            model_filename = f"models/{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
            self.predictor.save_model(model_filename)

            logger.info("Pipeline completed successfully!")
            logger.info(f"Test metrics: {test_metrics}")

            return test_metrics

        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise

    def compare_models(self, symbol, start_date, end_date, prediction_days=1):
        """
        Compare different ML models.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
            prediction_days (int): Days ahead to predict
        """
        try:
            logger.info("Starting model comparison...")

            # Fetch and preprocess data
            self.data = self.fetcher.fetch_data(symbol, start_date, end_date)
            self.processed_data, X_train, X_val, X_test, y_train, y_val, y_test = \
                self.preprocessor.preprocess_pipeline(self.data, prediction_days)

            # Initialize model comparison
            comparison = ModelComparison()

            # Add different models
            models_to_compare = ['random_forest', 'gradient_boosting', 'logistic', 'svm']
            for model_type in models_to_compare:
                comparison.add_model(model_type, model_type)

            # Train all models
            comparison.train_all(X_train, y_train, X_val, y_val)

            # Evaluate all models
            results = comparison.evaluate_all(X_test, y_test)

            # Plot comparison
            comparison.plot_comparison()

            # Get best model
            best_model = comparison.get_best_model()
            logger.info(f"Best performing model: {best_model}")

            # Set the best model as current predictor
            self.predictor = comparison.models[best_model]

            return results

        except Exception as e:
            logger.error(f"Error in model comparison: {str(e)}")
            raise

    def predict_next_day(self, symbol):
        """
        Predict next day's price movement.

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Prediction result
        """
        try:
            if self.predictor is None:
                raise ValueError("No trained model available. Run pipeline first.")

            # Get latest data (last 60 days for context)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=60)

            latest_data = self.fetcher.fetch_data(symbol, str(start_date), str(end_date))

            # Preprocess latest data
            _, latest_processed, _, _, _, _ = self.preprocessor.preprocess_pipeline(latest_data)

            # Make prediction
            result = self.predictor.predict_future(latest_processed)

            logger.info(f"Next day prediction for {symbol}: {result}")
            return result

        except Exception as e:
            logger.error(f"Error predicting next day: {str(e)}")
            raise

    def analyze_stock(self, symbol, start_date, end_date):
        """
        Perform comprehensive analysis of a stock.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
        """
        try:
            logger.info(f"Analyzing {symbol}...")

            # Fetch data
            data = self.fetcher.fetch_data(symbol, start_date, end_date)

            # Get company info
            company_info = self.fetcher.get_company_info(symbol)

            # Basic statistics
            print(f"\n=== Analysis for {symbol} ===")
            print(f"Company: {company_info.get('name', 'N/A')}")
            print(f"Sector: {company_info.get('sector', 'N/A')}")
            print(f"Industry: {company_info.get('industry', 'N/A')}")
            print(f"Data points: {len(data)}")
            print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            print(f"Average price: ${data['Close'].mean():.2f}")
            print(f"Price volatility (std): ${data['Close'].std():.2f}")
            print(f"Max price: ${data['High'].max():.2f}")
            print(f"Min price: ${data['Low'].min():.2f}")

            # Plot price history
            plt.figure(figsize=(12, 6))
            plt.plot(data['Date'], data['Close'], label='Close Price', linewidth=2)
            plt.title(f'{symbol} Price History')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            return data, company_info

        except Exception as e:
            logger.error(f"Error analyzing stock: {str(e)}")
            raise

    def load_model(self, model_path):
        """
        Load a saved model.

        Args:
            model_path (str): Path to saved model
        """
        try:
            if self.predictor is None:
                self.predictor = StockPredictor()

            self.predictor.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description='Stock Trading ML Application')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default='random_forest', choices=['random_forest', 'gradient_boosting', 'logistic', 'svm'], help='ML model type')
    parser.add_argument('--prediction-days', type=int, default=1, help='Days ahead to predict')
    parser.add_argument('--compare-models', action='store_true', help='Compare different models')
    parser.add_argument('--analyze', action='store_true', help='Perform stock analysis')
    parser.add_argument('--predict-next', action='store_true', help='Predict next day movement')
    parser.add_argument('--load-model', type=str, help='Load saved model path')

    args = parser.parse_args()

    # Initialize application
    app = StockTradingML()

    try:
        if args.load_model:
            app.load_model(args.load_model)

        if args.analyze:
            app.analyze_stock(args.symbol, args.start_date, args.end_date)

        if args.compare_models:
            app.compare_models(args.symbol, args.start_date, args.end_date, args.prediction_days)
        elif not args.analyze and not args.predict_next:
            # Run standard pipeline
            app.run_pipeline(args.symbol, args.start_date, args.end_date,
                           args.model, args.prediction_days)

        if args.predict_next:
            prediction = app.predict_next_day(args.symbol)
            print(f"\nNext day prediction: {prediction}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
