import pandas as pd
import numpy as np
import ta


def compute_ta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rsi14"] = ta.momentum.RSIIndicator(out["close"], window=14).rsi()
    out["ema50"] = ta.trend.EMAIndicator(out["close"], window=50).ema_indicator()
    out["ema200"] = ta.trend.EMAIndicator(out["close"], window=200).ema_indicator()
    out["atr14"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=14
    ).average_true_range()
    out["trend"] = np.where(out["ema50"] > out["ema200"], 1, -1)

    return out


def ta_signal(row) -> str:
    if row["rsi14"] < 30 and row["trend"] == 1:
        return "buy"

    if row["rsi14"] > 60 and row["trend"] == -1:
        return "sell"

    return "hold"
