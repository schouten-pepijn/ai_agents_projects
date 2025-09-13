import pandas as pd
from sklearn.linear_model import Ridge


# Simple rule-based signal: 1 if short-term SMA > long-term SMA, else 0
def rule_signal(feature: pd.DataFrame) -> pd.Series:
    return (feature["sma_10"] > feature["sma_50"]).astype(int)


# Machine learning-based signal using Ridge regression
def ml_signal(feature: pd.DataFrame) -> pd.Series:
    X = feature[["rsi_14", "sma_10", "sma_50", "bb_low", "bb_high"]]
    y = feature["ret_1d"].shift(-1)
    X, y = X.iloc[:-1], y.iloc[:-1]

    if len(X) < 60:
        return pd.Series(0, index=feature.index, dtype=int)

    split = int(len(X) * 0.7)
    model = Ridge(alpha=1.0).fit(X.iloc[:split], y.iloc[:split])

    preds = model.predict(X.iloc[split:], index=X.index[split:])

    signal = (preds > 0).astype(int)

    return signal.reindex(feature.index, method="ffill").fillna(0).astype(int)
