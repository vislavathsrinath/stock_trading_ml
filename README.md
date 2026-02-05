# Stock Trading ML - Real-Time Trading Dashboard

A comprehensive machine learning-powered stock and cryptocurrency trading analysis platform with **real-time data updates**.

## Features

### Real-Time Data Updates
- **Live Price Feeds**: Real-time stock prices via Alpha Vantage API
- **Cryptocurrency Prices**: Live crypto prices via CoinGecko API
- **Automatic Updates**: Data refreshes every 30 seconds for prices, 1 minute for charts
- **Fallback System**: Graceful fallback to Yahoo Finance when APIs are unavailable

### Machine Learning Models
- Random Forest, Gradient Boosting, and SVM models
- Automated model training and evaluation
- Real-time predictions for price movements

### Web Dashboard
- Interactive charts with Chart.js
- Real-time price updates
- Trading signal generation
- Backtesting capabilities
- Alert system with email/SMS notifications

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_trading_ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (optional, for enhanced real-time data):
```bash
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

## Usage

### Real-Time Web Application
Run the real-time trading dashboard:
```bash
python web_app_real_time.py
```
Open http://localhost:5000 in your browser.

### Test Real-Time Data
Test the real-time data fetching:
```bash
python test_real_time.py
```

### Traditional ML Pipeline
Run the standard ML training pipeline:
```bash
python main.py --symbol AAPL --analyze
```

## API Keys

For enhanced real-time data:

1. **Alpha Vantage** (Stocks): Get free API key at https://www.alphavantage.co/support/#api-key
2. **CoinGecko** (Crypto): No API key required
3. **Email/SMS Alerts**: Configure SMTP and Twilio credentials in environment variables

## Real-Time Data Sources

- **Stocks**: Alpha Vantage API (real-time) → Yahoo Finance (fallback)
- **Cryptocurrencies**: CoinGecko API (near real-time)
- **Commodities**: Yahoo Finance (delayed)

## Architecture

- `data_fetcher_real_time.py`: Real-time data fetching with multiple APIs
- `web_app_real_time.py`: Flask web application with live updates
- `model_training.py`: ML model training and prediction
- `alert_system.py`: Notification system for trading signals

## Data Update Intervals

- **Price Updates**: Every 30 seconds
- **Chart Data**: Every 1 minute
- **Background Refresh**: Continuous background updates

## Supported Assets

- **Stocks**: TSLA, SAP, AAPL, etc.
- **Crypto**: BTC-USD, ETH-USD, BNB-USD, etc.
- **Commodities**: Gold (GC=F), Silver (SI=F)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with real-time data
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
