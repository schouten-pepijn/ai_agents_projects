import pandas as pd
import ta


def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["adj_close"].pct_change()

    sma10 = ta.trend.SMAIndicator(close=out["adj_close"], window=10)
    sma50 = ta.trend.SMAIndicator(close=out["adj_close"], window=50)
    out["sma_10"] = sma10.sma_indicator()
    out["sma_50"] = sma50.sma_indicator()

    rsi14 = ta.momentum.RSIIndicator(close=out["adj_close"], window=14)
    out["rsi_14"] = rsi14.rsi()

    bb = ta.volatility.BollingerBands(close=out["adj_close"], window=20, window_dev=2)
    out["bb_low"] = bb.bollinger_lband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_high"] = bb.bollinger_hband()

    return out
