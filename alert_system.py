#!/usr/bin/env python3
"""
Alert System Module
Handles email and SMS notifications for trading signals and price alerts.
"""

import logging
import smtplib
import json
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AlertType:
    """Enumeration of alert types."""
    SIGNAL = "SIGNAL"
    PRICE_ALERT = "PRICE_ALERT"
    PORTFOLIO = "PORTFOLIO"
    SYSTEM = "SYSTEM"

class AlertPriority:
    """Enumeration of alert priorities."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Alert:
    """Represents an alert with metadata."""

    def __init__(self, alert_type, title, message, priority=AlertPriority.MEDIUM, metadata=None):
        """
        Initialize an alert.

        Args:
            alert_type (AlertType): Type of alert
            title (str): Alert title
            message (str): Alert message
            priority (AlertPriority): Alert priority
            metadata (dict): Additional metadata
        """
        self.alert_type = alert_type
        self.title = title
        self.message = message
        self.priority = priority
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
        self.id = f"{alert_type}_{int(self.timestamp.timestamp())}"

    def to_dict(self):
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'type': self.alert_type,
            'title': self.title,
            'message': self.message,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class EmailNotifier:
    """Handles email notifications."""

    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587, username=None, password=None):
        """
        Initialize email notifier.

        Args:
            smtp_server (str): SMTP server address
            smtp_port (int): SMTP server port
            username (str): Email username
            password (str): Email password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username or os.getenv('EMAIL_USERNAME')
        self.password = password or os.getenv('EMAIL_PASSWORD')

    def send_alert(self, alert, recipients):
        """
        Send alert via email.

        Args:
            alert (Alert): Alert to send
            recipients (list): List of email recipients

        Returns:
            bool: Success status
        """
        try:
            if not self.username or not self.password:
                logger.warning("Email credentials not configured")
                return False

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.priority}] {alert.title}"

            # Add body
            body = f"""
{alert.message}

Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Type: {alert.alert_type}
Priority: {alert.priority}

{json.dumps(alert.metadata, indent=2) if alert.metadata else ''}
            """
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.username, recipients, text)
            server.quit()

            logger.info(f"Email alert sent to {len(recipients)} recipients")
            return True

        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False

class SMSNotifier:
    """Handles SMS notifications via Twilio."""

    def __init__(self, account_sid=None, auth_token=None, from_number=None):
        """
        Initialize SMS notifier.

        Args:
            account_sid (str): Twilio account SID
            auth_token (str): Twilio auth token
            from_number (str): Twilio phone number
        """
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = from_number or os.getenv('TWILIO_FROM_NUMBER')

        # Import twilio only if available
        try:
            from twilio.rest import Client
            self.twilio_client = Client(self.account_sid, self.auth_token) if self.account_sid else None
        except ImportError:
            logger.warning("Twilio not installed. SMS notifications disabled.")
            self.twilio_client = None

    def send_alert(self, alert, recipients):
        """
        Send alert via SMS.

        Args:
            alert (Alert): Alert to send
            recipients (list): List of phone numbers

        Returns:
            bool: Success status
        """
        try:
            if not self.twilio_client:
                logger.warning("Twilio client not configured")
                return False

            message_body = f"{alert.title}: {alert.message}"

            success_count = 0
            for recipient in recipients:
                try:
                    message = self.twilio_client.messages.create(
                        body=message_body,
                        from_=self.from_number,
                        to=recipient
                    )
                    success_count += 1
                    logger.info(f"SMS sent to {recipient}")
                except Exception as e:
                    logger.error(f"Error sending SMS to {recipient}: {str(e)}")

            return success_count > 0

        except Exception as e:
            logger.error(f"Error sending SMS alert: {str(e)}")
            return False

class WebhookNotifier:
    """Handles webhook notifications."""

    def __init__(self, webhook_url=None):
        """
        Initialize webhook notifier.

        Args:
            webhook_url (str): Webhook URL for notifications
        """
        self.webhook_url = webhook_url or os.getenv('WEBHOOK_URL')

    def send_alert(self, alert, headers=None):
        """
        Send alert via webhook.

        Args:
            alert (Alert): Alert to send
            headers (dict): Additional headers

        Returns:
            bool: Success status
        """
        try:
            if not self.webhook_url:
                logger.warning("Webhook URL not configured")
                return False

            payload = alert.to_dict()
            default_headers = {'Content-Type': 'application/json'}
            if headers:
                default_headers.update(headers)

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=default_headers,
                timeout=10
            )

            if response.status_code == 200:
                logger.info("Webhook alert sent successfully")
                return True
            else:
                logger.error(f"Webhook failed with status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending webhook alert: {str(e)}")
            return False

class AlertManager:
    """Manages alerts and notification channels."""

    def __init__(self, config_file=None):
        """
        Initialize alert manager.

        Args:
            config_file (str): Path to configuration file
        """
        self.email_notifier = EmailNotifier()
        self.sms_notifier = SMSNotifier()
        self.webhook_notifier = WebhookNotifier()
        self.alert_history = []
        self.subscribers = {
            'email': [],
            'sms': [],
            'webhook': []
        }

        # Load configuration
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def load_config(self, config_file):
        """Load alert configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Configure notifiers
            if 'email' in config:
                email_config = config['email']
                self.email_notifier = EmailNotifier(**email_config)

            if 'sms' in config:
                sms_config = config['sms']
                self.sms_notifier = SMSNotifier(**sms_config)

            if 'webhook' in config:
                webhook_config = config['webhook']
                self.webhook_notifier = WebhookNotifier(**webhook_config)

            # Load subscribers
            if 'subscribers' in config:
                self.subscribers.update(config['subscribers'])

            logger.info("Alert configuration loaded")

        except Exception as e:
            logger.error(f"Error loading alert config: {str(e)}")

    def add_subscriber(self, channel, contact):
        """
        Add a subscriber to a notification channel.

        Args:
            channel (str): Notification channel ('email', 'sms', 'webhook')
            contact (str): Contact information (email, phone, etc.)
        """
        if channel in self.subscribers:
            if contact not in self.subscribers[channel]:
                self.subscribers[channel].append(contact)
                logger.info(f"Added {contact} to {channel} notifications")

    def remove_subscriber(self, channel, contact):
        """
        Remove a subscriber from a notification channel.

        Args:
            channel (str): Notification channel
            contact (str): Contact information
        """
        if channel in self.subscribers and contact in self.subscribers[channel]:
            self.subscribers[channel].remove(contact)
            logger.info(f"Removed {contact} from {channel} notifications")

    def create_signal_alert(self, signal_data):
        """
        Create an alert for a trading signal.

        Args:
            signal_data (dict): Signal data

        Returns:
            Alert: Created alert
        """
        symbol = signal_data['symbol']
        signal_type = signal_data['signal_type']
        strength = signal_data['strength']
        confidence = signal_data['confidence']

        title = f"{signal_type} Signal: {symbol}"
        message = f"""
Trading signal generated for {symbol}:
- Signal: {signal_type}
- Strength: {strength}
- Confidence: {confidence:.1%}
- Price: ${signal_data.get('price', 'N/A')}

Risk Assessment:
- Risk Level: {signal_data.get('risk_assessment', {}).get('risk_level', 'N/A')}
- Stop Loss: ${signal_data.get('risk_assessment', {}).get('stop_loss', 'N/A')}
- Take Profit: ${signal_data.get('risk_assessment', {}).get('take_profit', 'N/A')}
        """

        priority = AlertPriority.HIGH if strength == 'STRONG' else AlertPriority.MEDIUM

        alert = Alert(
            alert_type=AlertType.SIGNAL,
            title=title,
            message=message.strip(),
            priority=priority,
            metadata=signal_data
        )

        return alert

    def create_price_alert(self, symbol, current_price, target_price, condition):
        """
        Create an alert for price target reached.

        Args:
            symbol (str): Asset symbol
            current_price (float): Current price
            target_price (float): Target price
            condition (str): Alert condition ('above', 'below')

        Returns:
            Alert: Created alert
        """
        title = f"Price Alert: {symbol}"
        message = f"""
Price alert triggered for {symbol}:
- Current Price: ${current_price:.2f}
- Target Price: ${target_price:.2f}
- Condition: Price went {condition} target

This is an automated price alert.
        """

        alert = Alert(
            alert_type=AlertType.PRICE_ALERT,
            title=title,
            message=message.strip(),
            priority=AlertPriority.MEDIUM,
            metadata={
                'symbol': symbol,
                'current_price': current_price,
                'target_price': target_price,
                'condition': condition
            }
        )

        return alert

    def create_portfolio_alert(self, portfolio_data, alert_type):
        """
        Create an alert for portfolio events.

        Args:
            portfolio_data (dict): Portfolio data
            alert_type (str): Type of portfolio alert

        Returns:
            Alert: Created alert
        """
        title = f"Portfolio Alert: {alert_type.title()}"
        message = f"Portfolio alert: {alert_type}\n{json.dumps(portfolio_data, indent=2)}"

        alert = Alert(
            alert_type=AlertType.PORTFOLIO,
            title=title,
            message=message,
            priority=AlertPriority.MEDIUM,
            metadata=portfolio_data
        )

        return alert

    def send_alert(self, alert, channels=None):
        """
        Send an alert through specified channels.

        Args:
            alert (Alert): Alert to send
            channels (list): List of channels to use (default: all configured)

        Returns:
            dict: Send results by channel
        """
        if channels is None:
            channels = []
            if self.subscribers['email']:
                channels.append('email')
            if self.subscribers['sms']:
                channels.append('sms')
            if self.webhook_notifier.webhook_url:
                channels.append('webhook')

        results = {}

        # Send via email
        if 'email' in channels and self.subscribers['email']:
            results['email'] = self.email_notifier.send_alert(alert, self.subscribers['email'])

        # Send via SMS
        if 'sms' in channels and self.subscribers['sms']:
            results['sms'] = self.sms_notifier.send_alert(alert, self.subscribers['sms'])

        # Send via webhook
        if 'webhook' in channels:
            results['webhook'] = self.webhook_notifier.send_alert(alert)

        # Store alert in history
        self.alert_history.append(alert.to_dict())

        # Log results
        success_count = sum(1 for r in results.values() if r)
        logger.info(f"Alert sent via {success_count}/{len(results)} channels")

        return results

    def get_alert_history(self, limit=100):
        """Get recent alert history."""
        return self.alert_history[-limit:] if limit else self.alert_history

    def clear_history(self):
        """Clear alert history."""
        self.alert_history.clear()
        logger.info("Alert history cleared")

# Global alert manager instance
alert_manager = AlertManager()

def quick_signal_alert(signal_data):
    """
    Quick function to send a signal alert.

    Args:
        signal_data (dict): Signal data
    """
    alert = alert_manager.create_signal_alert(signal_data)
    alert_manager.send_alert(alert)

def quick_price_alert(symbol, current_price, target_price, condition):
    """
    Quick function to send a price alert.

    Args:
        symbol (str): Asset symbol
        current_price (float): Current price
        target_price (float): Target price
        condition (str): Alert condition
    """
    alert = alert_manager.create_price_alert(symbol, current_price, target_price, condition)
    alert_manager.send_alert(alert)
