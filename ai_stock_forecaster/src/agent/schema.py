from typing import TypedDict, Literal
import pandas as pd


class State(TypedDict, total=False):
    question: str
    symbol: str
    provider: Literal["yf"]
    data: pd.DataFrame
    features: pd.DataFrame
    signals: dict
    results: dict
