from langgraph.graph import StateGraph, END
from .schema import State
from .nodes import route_question, fetch, featurize, forecast, backtest


def build_graph():
    g = StateGraph(State)

    g.add_node("fetch", fetch)
    g.add_node("featurize", featurize)
    g.add_node("forecast", forecast)
    g.add_node("backtest", backtest)

    g.set_entry_point(route_question)
    g.add_edge("fetch", "featurize")
    g.add_edge("featurize", "forecast")
    g.add_edge("forecast", "backtest")
    g.add_edge("backtest", END)

    return g.compile()
