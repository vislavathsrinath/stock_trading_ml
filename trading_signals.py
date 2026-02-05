"""
Trading Signals Module
Generates trading signals using ML predictions and technical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class TradingSignal:
    """Represents a trading signal with confidence."""

    def __init__(self, signal: str, confidence: float, details: Optional[Dict[str, Any]] = None):
        self.signal = signal.upper()  # BUY, SELL, HOLD
        self.confidence = confidence
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'signal': self.signal,
            'confidence': self.confidence,
            'details': self.details
        }

class SignalGenerator:
    """Generates trading signals from ML predictions and technical indicators."""

    def __init__(self):
        """Initialize the signal generator."""
        self.signal_threshold = 0.6  # Minimum confidence for action signals

    def generate_signal_from_prediction(self, prediction: Dict[str, Any]) -> TradingSignal:
        """
        Generate trading signal from ML prediction.

        Args:
            prediction: ML model prediction results

        Returns:
            TradingSignal object
        """
        try:
            # Extract prediction data
            predicted_price = prediction.get('predicted_price', 0)
            current_price = prediction.get('current_price', 0)
            confidence = prediction.get('confidence', 0.5)

            if predicted_price > current_price * 1.02:  # 2% increase predicted
                signal = 'BUY'
            elif predicted_price < current_price * 0.98:  # 2% decrease predicted
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return TradingSignal(signal, confidence, {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'source': 'ml_prediction'
            })

        except Exception as e:
            return TradingSignal('HOLD', 0.5, {'error': str(e), 'source': 'ml_prediction'})

    def generate_technical_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal from technical analysis.

        Args:
            data: Historical price data

        Returns:
            TradingSignal object
        """
        try:
            if data.empty or len(data) < 20:
                return TradingSignal('HOLD', 0.5, {'error': 'Insufficient data', 'source': 'technical'})

            # Simple moving averages
            data = data.copy()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest

            # Basic technical signals
            if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
                signal = 'BUY'  # Golden cross
                confidence = 0.7
            elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
                signal = 'SELL'  # Death cross
                confidence = 0.7
            else:
                signal = 'HOLD'
                confidence = 0.5

            return TradingSignal(signal, confidence, {
                'sma_20': latest['SMA_20'],
                'sma_50': latest['SMA_50'],
                'source': 'technical'
            })

        except Exception as e:
            return TradingSignal('HOLD', 0.5, {'error': str(e), 'source': 'technical'})

    def combine_signals(self, ml_signal: TradingSignal, tech_signal: TradingSignal) -> TradingSignal:
        """
        Combine ML and technical signals into a final signal.

        Args:
            ml_signal: Signal from ML prediction
            tech_signal: Signal from technical analysis

        Returns:
            Combined TradingSignal object
        """
        try:
            # Weight the signals (ML gets higher weight)
            ml_weight = 0.6
            tech_weight = 0.4

            # Convert signals to numerical values
            signal_values = {'BUY': 1, 'HOLD': 0, 'SELL': -1}

            ml_value = signal_values.get(ml_signal.signal, 0) * ml_signal.confidence
            tech_value = signal_values.get(tech_signal.signal, 0) * tech_signal.confidence

            # Weighted combination
            combined_value = ml_value * ml_weight + tech_value * tech_weight
            combined_confidence = (ml_signal.confidence * ml_weight + tech_signal.confidence * tech_weight)

            # Determine final signal
            if combined_value > 0.3:
                final_signal = 'BUY'
            elif combined_value < -0.3:
                final_signal = 'SELL'
            else:
                final_signal = 'HOLD'

            # Combine details
            combined_details = {
                'ml_signal': ml_signal.to_dict(),
                'tech_signal': tech_signal.to_dict(),
                'combined_value': combined_value,
                'source': 'combined'
            }

            return TradingSignal(final_signal, combined_confidence, combined_details)

        except Exception as e:
            return TradingSignal('HOLD', 0.5, {'error': str(e), 'source': 'combined'})
