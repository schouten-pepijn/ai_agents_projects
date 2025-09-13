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

    # Handle NaN values by dropping rows with any missing values
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    if len(X_clean) < 60:
        return pd.Series(0, index=feature.index, dtype=int)

    split = int(len(X_clean) * 0.7)
    model = Ridge(alpha=1.0).fit(X_clean.iloc[:split], y_clean.iloc[:split])

    # Make predictions on the clean data
    preds = model.predict(X_clean.iloc[split:])

    # Create a series with predictions aligned to the original index
    pred_series = pd.Series(0, index=feature.index, dtype=float)
    pred_series.loc[X_clean.index[split:]] = preds

    signal = (pred_series > 0).astype(int)

    return signal
