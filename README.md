# Stock Trading ML - Real-Time Trading Dashboard

Hey there! 👋 Welcome to my personal stock trading assistant. This is a cool Python project I built that helps me track stocks, crypto, and commodities in real-time, plus uses AI to predict price movements. Think of it as your own trading dashboard that never sleeps!

## What This Project Does

Imagine having a personal assistant that:
- **Tracks real-time prices** for Bitcoin, Tesla, Gold, Silver, BONK crypto, and SAP stock
- **Shows beautiful charts** with price trends and volume data
- **Uses AI to predict** if prices will go up or down tomorrow
- **Updates automatically** every 5 minutes with fresh data
- **Works on your phone or computer** - it's fully responsive!

## Quick Start - Let's Get This Running!

### Step 1: Install the Stuff You Need
First, make sure you have Python installed (I used Python 3.8+). Then run this in your terminal:

```bash
cd stock_trading_ml
pip install -r requirements.txt
```

This installs all the libraries we need like yfinance for stock data, scikit-learn for AI models, flask for the web app, and more.

### Step 2: Fire Up the Dashboard
Ready to see your trading dashboard? Just run:

```bash
python web_app.py
```

Boom! Open your browser and go to `http://localhost:5000`. You'll see your real-time trading dashboard with all the assets updating live!

## What You'll See on the Dashboard

### Main Dashboard Page
- **Live price cards** for all 6 assets (BTC, Gold, Silver, BONK, TSLA, SAP)
- **Color-coded changes** - green for gains, red for losses
- **AI prediction badges** showing what the computer thinks will happen next
- **Mini charts** showing the last 30 days of price action
- **Auto-refresh** every 5 minutes so you always have fresh data

### Individual Asset Pages
Click on any asset card to see the detailed view:
- **90-day price history** with interactive zoomable charts
- **Technical indicators** like RSI, MACD, moving averages
- **Volume analysis** showing trading activity
- **Price statistics** including 52-week highs/lows
- **AI confidence scores** for the predictions

## Playing Around with the Code

### Command Line Fun
Want to analyze stocks without the web interface? Try these commands:

```bash
# Get basic info about Apple stock
python main.py --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31 --analyze

# Train an AI model to predict Tesla stock movements
python main.py --symbol TSLA --start-date 2020-01-01 --end-date 2023-12-31 --model random_forest

# Compare different AI models to see which is best
python main.py --symbol TSLA --start-date 2020-01-01 --end-date 2023-12-31 --compare-models
```

### Interactive Notebook
For the full data science experience, open the Jupyter notebook:

```bash
jupyter notebook stock_analysis.ipynb
```

This lets you run code cells interactively, create your own charts, and experiment with different stocks and time periods.

## How the AI Prediction Works

I trained machine learning models using historical stock data and technical indicators. The system looks at:

- **Price changes** over different time periods
- **Technical indicators** like RSI, MACD, moving averages
- **Volume data** showing trading activity
- **Bollinger Bands** for volatility analysis

The AI then predicts whether the price will go UP or DOWN the next day, with a confidence score. It's not perfect (no crystal ball is!), but it's a fun way to see what the data suggests.

## The Tech Behind It

### Backend (The Brain)
- **Flask web framework** - handles the web server and API calls
- **Yahoo Finance API** - pulls real-time and historical stock data
- **Scikit-learn** - the machine learning library for predictions
- **Pandas & NumPy** - data crunching and analysis
- **Background threads** - keep data fresh without blocking the UI

### Frontend (What You See)
- **Bootstrap** - makes it look good and work on mobile
- **Chart.js** - creates beautiful, interactive charts
- **AJAX calls** - updates data in real-time without page refreshes
- **Responsive design** - looks great on phones, tablets, and desktops

## Project Files Overview

```
stock_trading_ml/
├── main.py                 # Command-line interface for analysis
├── web_app.py             # Flask web application (the dashboard!)
├── data_fetcher.py        # Grabs stock data from Yahoo Finance
├── data_preprocessing.py  # Cleans data and adds technical indicators
├── model_training.py      # Trains and runs the AI prediction models
├── stock_analysis.ipynb   # Interactive Jupyter notebook
├── requirements.txt       # List of Python libraries needed
├── templates/             # HTML templates for the web pages
│   ├── index.html        # Main dashboard page
│   └── asset.html        # Individual asset detail pages
└── models/               # Saved AI models (created when you train)
```

## Adding Your Own Assets

Want to track different stocks or crypto? Edit the `TRACKED_ASSETS` in `web_app.py`:

```python
TRACKED_ASSETS = {
    'stocks': ['TSLA', 'AAPL', 'GOOGL'],  # Add your favorite stocks
    'crypto': ['BTC-USD', 'ETH-USD'],     # Add more crypto
    'commodities': ['GC=F', 'SI=F']       # Gold, silver, etc.
}
```

## Tips for Getting Started

1. **Start with the web dashboard** - it's the easiest way to see everything working
2. **Try different stocks** - see how the AI predictions vary
3. **Check the notebook** - great for learning how the analysis works
4. **Experiment with dates** - see how predictions change over different time periods
5. **Don't take it too seriously** - this is for learning and fun, not financial advice!

## Important Disclaimer

⚠️ **This is for educational purposes only!**

- Past performance doesn't predict future results
- AI predictions can be wrong (that's why they're called predictions!)
- Always do your own research before making investment decisions
- I'm not a financial advisor, and this isn't financial advice
- Trade at your own risk, and only with money you can afford to lose

## What's Next?

This project is always evolving! Here are some ideas for the future:
- Add more technical indicators
- Include news sentiment analysis
- Build a mobile app version
- Add portfolio tracking features
- Implement automated trading signals
- Add more machine learning models

## Questions or Issues?

Found a bug? Have a suggestion? Want to contribute? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Reach out with questions

Happy trading, and may your predictions be profitable! 📈🚀

---

*Built with ❤️ using Python, Flask, and a whole lot of curiosity about markets*
