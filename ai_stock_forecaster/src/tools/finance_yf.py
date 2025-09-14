import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def get_daily(symbol: str, period="max", interval="1d") -> pd.DataFrame:
    """Fetch daily price data for a symbol.

    Accepts standard yfinance `period` values (e.g., '1y', '5y', 'max') and
    additionally supports '15y' and '20y' by converting them to explicit start
    dates and calling `history(start=..., end=...)`.
    """
    ticker = yf.Ticker(symbol)

    # Handle extended multi-year periods not directly supported by yfinance period
    if period in ("15y", "20y"):
        years = int(period.replace("y", ""))
        end = datetime.utcnow().date()
        start = end - timedelta(days=365 * years)
        df = ticker.history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            auto_adjust=True,
        )
    else:
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
