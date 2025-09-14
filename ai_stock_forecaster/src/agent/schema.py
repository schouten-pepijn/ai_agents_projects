from typing import TypedDict, Literal
import pandas as pd


class State(TypedDict, total=False):
    question: str
    symbol: str
    provider: Literal["yf"]
    period: str  # Added for time span selection (1y, 2y, 5y, 10y, max)
    data: pd.DataFrame
    features: pd.DataFrame
    signals: dict
    results: dict
