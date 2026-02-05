#!/usr/bin/env python3
"""
Data Preprocessing Module
Handles cleaning, feature engineering, and preparing data for ML models.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class StockDataPreprocessor:
    """Class for preprocessing stock data."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = None
        self.feature_columns = None

    def clean_data(self, data):
        """
        Clean the raw stock data.

        Args:
            data (pd.DataFrame): Raw stock data

        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            logger.info("Cleaning stock data...")

            # Remove rows with missing values
            cleaned_data = data.dropna()

            # Remove duplicate dates
            cleaned_data = cleaned_data.drop_duplicates(subset=['Date'])

            # Sort by date
            cleaned_data = cleaned_data.sort_values('Date').reset_index(drop=True)

            # Ensure numeric columns are float
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in cleaned_data.columns:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')

            # Remove any remaining NaN values after conversion
            cleaned_data = cleaned_data.dropna()

            logger.info(f"Data cleaned. Shape: {cleaned_data.shape}")
            return cleaned_data

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def add_technical_indicators(self, data):
        """
        Add technical indicators as features.

        Args:
            data (pd.DataFrame): Cleaned stock data

        Returns:
            pd.DataFrame: Data with technical indicators
        """
        try:
            logger.info("Adding technical indicators...")

            df = data.copy()

            # Simple Moving Averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()

            # Exponential Moving Averages
            df['EMA_5'] = df['Close'].ewm(span=5).mean()
            df['EMA_10'] = df['Close'].ewm(span=10).mean()

            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Moving Average Convergence Divergence (MACD)
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)

            # Price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5'] = df['Close'].pct_change(5)

            # Volume indicators
            df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']

            # Drop NaN values created by rolling calculations
            df = df.dropna().reset_index(drop=True)

            logger.info(f"Technical indicators added. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise

    def create_target_variable(self, data, prediction_days=1):
        """
        Create target variable for prediction.

        Args:
            data (pd.DataFrame): Data with features
            prediction_days (int): Number of days ahead to predict

        Returns:
            pd.DataFrame: Data with target variable
        """
        try:
            logger.info(f"Creating target variable for {prediction_days} day(s) ahead...")

            df = data.copy()

            # Create target: 1 if price goes up, 0 if down
            df['Target'] = (df['Close'].shift(-prediction_days) > df['Close']).astype(int)

            # Remove rows where target is NaN (last prediction_days rows)
            df = df[:-prediction_days].reset_index(drop=True)

            logger.info(f"Target variable created. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
            raise

    def scale_features(self, data, feature_columns=None, method='standard'):
        """
        Scale numerical features.

        Args:
            data (pd.DataFrame): Data to scale
            feature_columns (list): Columns to scale (if None, auto-detect)
            method (str): Scaling method ('standard' or 'minmax')

        Returns:
            pd.DataFrame: Scaled data
        """
        try:
            logger.info(f"Scaling features using {method} scaler...")

            df = data.copy()

            if feature_columns is None:
                # Auto-detect numerical columns (exclude Date and Target)
                exclude_cols = ['Date', 'Target']
                feature_columns = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

            self.feature_columns = feature_columns

            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Method must be 'standard' or 'minmax'")

            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])

            logger.info(f"Features scaled. Columns: {len(feature_columns)}")
            return df

        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def split_data(self, data, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets.

        Args:
            data (pd.DataFrame): Preprocessed data
            test_size (float): Test set proportion
            val_size (float): Validation set proportion
            random_state (int): Random state for reproducibility

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            logger.info("Splitting data into train/val/test sets...")

            # Features and target
            feature_cols = [col for col in data.columns if col not in ['Date', 'Target']]
            X = data[feature_cols]
            y = data['Target']

            # First split: train and temp (val+test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=(test_size + val_size), random_state=random_state, shuffle=False
            )

            # Second split: val and test
            val_ratio = val_size / (test_size + val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, shuffle=False
            )

            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def preprocess_pipeline(self, data, prediction_days=1, scale_method='standard'):
        """
        Complete preprocessing pipeline.

        Args:
            data (pd.DataFrame): Raw stock data
            prediction_days (int): Days ahead to predict
            scale_method (str): Feature scaling method

        Returns:
            tuple: (processed_data, X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            logger.info("Starting preprocessing pipeline...")

            # Clean data
            cleaned_data = self.clean_data(data)

            # Add technical indicators
            data_with_indicators = self.add_technical_indicators(cleaned_data)

            # Create target variable
            data_with_target = self.create_target_variable(data_with_indicators, prediction_days)

            # Scale features
            scaled_data = self.scale_features(data_with_target, method=scale_method)

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(scaled_data)

            logger.info("Preprocessing pipeline completed successfully")
            return scaled_data, X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
