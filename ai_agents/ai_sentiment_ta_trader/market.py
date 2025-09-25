import yfinance as yf
import pandas as pd


def fetch_bars(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]

    else:
        df.columns = [str(col).lower() for col in df.columns]

    df = df.reset_index().rename(columns={"Date": "date"})

    expected = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in expected if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[expected]
