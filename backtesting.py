import backtesting as bt
import pandas as pd

class Backtester:
    """Simple wrapper around the `backtesting` package to provide a
    `run_backtest(data, strategy, initial_capital)` method expected by the
    project.
    """

    def run_backtest(self, data, strategy, initial_capital=10000):
        try:
            # Ensure DataFrame index is datetime if it has a Date column
            if 'Date' in data.columns:
                df = data.copy()
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                df = data

            # `strategy` may be a class or an instance; backtesting requires a class
            strat = strategy if isinstance(strategy, type) else type(strategy)

            bt_instance = bt.Backtest(df, strat, cash=initial_capital)
            stats = bt_instance.run()

            # Convert stats to dict when possible
            try:
                return stats._asdict()
            except Exception:
                return dict(stats)
        except Exception as e:
            return {'error': str(e)}
