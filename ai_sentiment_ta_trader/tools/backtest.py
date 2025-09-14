from backtesting import Backtest, Strategy
import pandas as pd
import ta


def to_bt(df: pd.DataFrame):
    d = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).copy()
    d.set_index("Date", inplace=True)

    d["RSI"] = ta.momentum.RSIIndicator(d["Close"], window=14).rsi()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    return d


class RsiTrend(Strategy):
    def init(self):
        pass

    def next(self):
        rsi = self.data.RSI[-1]
        ma50 = self.data.MA50[-1]
        ma200 = self.data.MA200[-1]
        close = self.data.Close[-1]
        trend = 1 if ma50 > ma200 else -1

        if (rsi < 30) and trend == 1 and not self.position:
            self.buy(size=0.99)

        if self.position and (close < ma50 or rsi > 60):
            self.position.close()


def run_backtest(df):
    bt = Backtest(
        to_bt(df), RsiTrend, cash=10000, commission=0.001, trade_on_close=True
    )
    stats = bt.run()

    return stats.to_dict()
