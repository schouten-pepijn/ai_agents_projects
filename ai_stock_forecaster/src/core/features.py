import pandas as pd
import ta


def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # The get_daily function already provides clean column names
    # Column names are: ["open", "high", "low", "adj_close", "volume"]
    close_col = "adj_close"
    high_col = "high"
    low_col = "low"
    open_col = "open"
    volume_col = "volume"

    out["ret_1d"] = out[close_col].pct_change()  # Price-based features
    out["high_low_pct"] = (out[high_col] - out[low_col]) / out[close_col]
    out["close_open_pct"] = (out[close_col] - out[open_col]) / out[open_col]

    # Moving averages
    sma10 = ta.trend.SMAIndicator(close=out[close_col], window=10)
    sma50 = ta.trend.SMAIndicator(close=out[close_col], window=50)
    sma200 = ta.trend.SMAIndicator(close=out[close_col], window=200)
    out["sma_10"] = sma10.sma_indicator()
    out["sma_50"] = sma50.sma_indicator()
    out["sma_200"] = sma200.sma_indicator()

    # EMA
    ema12 = ta.trend.EMAIndicator(close=out[close_col], window=12)
    ema26 = ta.trend.EMAIndicator(close=out[close_col], window=26)
    out["ema_12"] = ema12.ema_indicator()
    out["ema_26"] = ema26.ema_indicator()

    # RSI
    rsi14 = ta.momentum.RSIIndicator(close=out[close_col], window=14)
    out["rsi_14"] = rsi14.rsi()

    # MACD
    macd = ta.trend.MACD(close=out[close_col])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=out[close_col], window=20, window_dev=2)
    out["bb_low"] = bb.bollinger_lband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_high"] = bb.bollinger_hband()
    out["bb_width"] = (out["bb_high"] - out["bb_low"]) / out["bb_mid"]
    out["bb_position"] = (out[close_col] - out["bb_low"]) / (
        out["bb_high"] - out["bb_low"]
    )

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=out[high_col], low=out[low_col], close=out[close_col]
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    williams = ta.momentum.WilliamsRIndicator(
        high=out[high_col], low=out[low_col], close=out[close_col]
    )
    out["williams_r"] = williams.williams_r()

    # Volume indicators (if volume is available)
    if volume_col in out.columns and not out[volume_col].isna().all():
        # Volume SMA (manual calculation)
        out["vol_sma"] = out[volume_col].rolling(window=20).mean()

        # On Balance Volume
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=out[close_col], volume=out[volume_col]
        )
        out["obv"] = obv.on_balance_volume()

        # Volume Rate of Change
        out["vol_roc"] = out[volume_col].pct_change()

        # Money Flow Index
        mfi = ta.volume.MFIIndicator(
            high=out[high_col],
            low=out[low_col],
            close=out[close_col],
            volume=out[volume_col],
        )
        out["mfi"] = mfi.money_flow_index()
    else:
        # Fill with zeros if no volume data
        out["vol_sma"] = 0
        out["obv"] = 0
        out["vol_roc"] = 0
        out["mfi"] = 0

    # ATR (Average True Range)
    atr = ta.volatility.AverageTrueRange(
        high=out[high_col], low=out[low_col], close=out[close_col]
    )
    out["atr"] = atr.average_true_range()

    # Price momentum features
    out["momentum_5"] = out[close_col].pct_change(5)
    out["momentum_10"] = out[close_col].pct_change(10)
    out["momentum_20"] = out[close_col].pct_change(20)

    # Volatility features
    out["volatility_10"] = out["ret_1d"].rolling(window=10).std()
    out["volatility_30"] = out["ret_1d"].rolling(window=30).std()

    # Price position relative to recent highs/lows
    out["high_52w"] = out[high_col].rolling(window=252).max()
    out["low_52w"] = out[low_col].rolling(window=252).min()
    out["position_52w"] = (out[close_col] - out["low_52w"]) / (
        out["high_52w"] - out["low_52w"]
    )

    return out
