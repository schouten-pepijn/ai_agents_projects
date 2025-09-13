import pandas as pd
import yfinance as yf


def get_daily(symbol: str, period="max", interval="1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise RuntimeError(
            f"No data found for symbol: {symbol} with period: {period} and interval: {interval} and yFinance"
        )

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "adj_close",
            "Volume": "volume",
        }
    )

    return df[
        [
            "open",
            "high",
            "low",
            "adj_close",
            "volume",
        ]
    ].sort_index()
