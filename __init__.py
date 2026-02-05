"""
Stock Trading ML Package
A comprehensive toolkit for stock market analysis and prediction using machine learning.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Stock market analysis and prediction using ML algorithms"

from .data_fetcher import StockDataFetcher
from .data_preprocessing import StockDataPreprocessor
from .model_training import StockPredictor, ModelComparison
from .main import StockTradingML

__all__ = [
    'StockDataFetcher',
    'StockDataPreprocessor',
    'StockPredictor',
    'ModelComparison',
    'StockTradingML'
]
