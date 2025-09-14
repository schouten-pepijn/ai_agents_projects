from .schema import State
from src.tools.finance_yf import get_daily
from src.core.features import technical_indicators
from src.core.models import ml_signal
from src.core.backtest import run_backtest
from src.core.report import summarize


# currently only supports Yahoo Finance
def route_question(state: State) -> State:
    q = (state.get("question") or "").lower()

    state["symbol"] = state.get("symbol") or (
        "AAPL"
        if "apple" in q
        else "MSFT" if "microsoft" in q else "NVDA" if "nvidia" in q else "AAPL"
    )
    state["provider"] = "yf"

    return state


def fetch(state: State) -> State:
    symbol = state["symbol"]
    period = state.get("period", "max")  # Default to "max" if not specified

    try:
        df = get_daily(symbol, period=period)

    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {e}") from e

    state["data"] = df

    return state


def featurize(state: State) -> State:
    state["features"] = technical_indicators(state["data"])

    return state


def forecast(state: State) -> State:
    features = state["features"]

    # Generate ML signals and get rule signals aligned to same prediction range
    ml_signals, rule_signals_aligned = ml_signal(features, return_aligned_rule=True)

    state["signals"] = {"rule": rule_signals_aligned, "ml": ml_signals}

    return state


def backtest(state: State) -> State:
    # Avoid column overlap by using features (which already contains adj_close)
    px = state["features"]

    r_rule = run_backtest(px, state["signals"]["rule"])
    r_ml = run_backtest(px, state["signals"]["ml"])

    state["results"] = summarize(state["symbol"], r_rule, r_ml)

    return state
