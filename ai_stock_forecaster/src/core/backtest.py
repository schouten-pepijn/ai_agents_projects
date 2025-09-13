import numpy as np
import pandas as pd


def run_backtest(prices: pd.DataFrame, signal: pd.Series, fee_bps: float = 1.0) -> dict:
    signal = signal.reindex(prices.index).fillna(0)

    ret = prices["adj_close"].pct_change().fillna(0)
    position = signal.shift(1).fillna(0)

    gross = position * ret
    turns = position.diff().abs().fillna(0)
    cost = turns * (fee_bps / 10000.0)
    equity = (1 + gross - cost).cumprod()

    stats = {
        "CAGR": (
            float((equity.iloc[-1]) ** (252 / max(1, len(equity))) - 1)
            if len(equity)
            else 0.0
        ),
        "Sharpe": (
            float((gross.mean() / gross.std()) * np.sqrt(252))
            if gross.std() > 0
            else 0.0
        ),
        "MaxDD": float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0,
        "Trades": int(turns.sum()),
    }

    return {"equity": equity, "stats": stats}
