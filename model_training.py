#!/usr/bin/env python3
"""
Model Training Module
Handles training and evaluation of ML models for stock prediction.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class StockPredictor:
    """Class for training and predicting stock price movements."""

    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor.

        Args:
            model_type (str): Type of ML model to use
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features (optional)
            y_val (pd.Series): Validation target (optional)
        """
        try:
            logger.info(f"Training {self.model_type} model...")

            # Train the model
            self.model.fit(X_train, y_train)

            # Calculate feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=X_train.columns
                ).sort_values(ascending=False)

            # Evaluate on training data
            train_pred = self.model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_pred)

            logger.info(f"Training completed. Train metrics: {train_metrics}")

            # Evaluate on validation data if provided
            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred)
                logger.info(f"Validation metrics: {val_metrics}")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X):
        """
        Make predictions.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            tuple: (predictions, probabilities)
        """
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive class

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_future(self, latest_data):
        """
        Predict future price movement based on latest data.

        Args:
            latest_data (pd.DataFrame): Latest processed data

        Returns:
            dict: Prediction result
        """
        try:
            # Use the most recent data point
            latest_features = latest_data.iloc[-1:].drop(['Date', 'Target'], axis=1, errors='ignore')

            prediction, probability = self.predict(latest_features)

            result = {
                'prediction': int(prediction[0]),
                'probability': float(probability[0]),
                'direction': 'UP' if prediction[0] == 1 else 'DOWN',
                'confidence': 'HIGH' if abs(probability[0] - 0.5) > 0.2 else 'MEDIUM'
            }

            return result

        except Exception as e:
            logger.error(f"Error predicting future: {str(e)}")
            raise

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info("Evaluating model...")

            predictions, probabilities = self.predict(X_test)
            metrics = self._calculate_metrics(y_test, predictions, probabilities)

            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))

            # Plot confusion matrix
            self._plot_confusion_matrix(y_test, predictions)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def _calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate evaluation metrics.

        Args:
            y_true (pd.Series): True labels
            y_pred (np.array): Predicted labels
            y_prob (np.array): Predicted probabilities (optional)

        Returns:
            dict: Metrics dictionary
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except Exception:
                metrics['roc_auc'] = None

        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            plt.title(f'Confusion Matrix - {self.model_type}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        except Exception as e:
            logger.warning(f"Could not plot confusion matrix: {str(e)}")

    def save_model(self, filepath):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_feature_importance(self, top_n=10):
        """
        Get feature importance if available.

        Args:
            top_n (int): Number of top features to return

        Returns:
            pd.Series: Feature importance
        """
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n)
        else:
            logger.warning("Feature importance not available for this model type")
            return None

class ModelComparison:
    """Class for comparing multiple ML models."""

    def __init__(self):
        """Initialize model comparison."""
        self.models = {}
        self.results = {}

    def add_model(self, name, model_type):
        """
        Add a model to compare.

        Args:
            name (str): Name for the model
            model_type (str): Type of ML model
        """
        self.models[name] = StockPredictor(model_type)

    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features (optional)
            y_val (pd.Series): Validation target (optional)
        """
        logger.info("Training all models...")

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.train(X_train, y_train, X_val, y_val)

    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all models.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            dict: Results for all models
        """
        logger.info("Evaluating all models...")

        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            self.results[name] = model.evaluate(X_test, y_test)

        return self.results

    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model.

        Args:
            metric (str): Metric to use for comparison

        Returns:
            str: Name of the best model
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all first.")

        best_model = max(self.results.items(), key=lambda x: x[1].get(metric, 0))
        return best_model[0]

    def plot_comparison(self, metrics=['accuracy', 'precision', 'recall', 'f1_score']):
        """
        Plot comparison of models.

        Args:
            metrics (list): Metrics to plot
        """
        try:
            if not self.results:
                raise ValueError("No evaluation results available. Run evaluate_all first.")

            # Prepare data for plotting
            model_names = list(self.results.keys())
            metric_data = {metric: [] for metric in metrics}

            for model_name in model_names:
                for metric in metrics:
                    value = self.results[model_name].get(metric, 0)
                    metric_data[metric].append(value)

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()

            for i, metric in enumerate(metrics):
                if i < len(axes):
                    axes[i].bar(model_names, metric_data[metric])
                    axes[i].set_title(f'{metric.capitalize()} Comparison')
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting comparison: {str(e)}")
