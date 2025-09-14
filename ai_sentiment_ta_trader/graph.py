import json

from typing import List, TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import config, settings
from market import fetch_bars
from news import fetch_news
from sentiment import sentiment_vader
from ta_tools import compute_ta, ta_signal
from prompts import FUSE_PROMPT


class State(TypedDict):
    symbol: str
    news: List[dict]
    ta_df_json: str
    sentiment: dict
    fuse_json: str


llm = ChatOllama(
    model=config.ollama_model, base_url=config.ollama_base_url, temperature=0.1
)


def node_fetch_news(state: State) -> State:
    query = f"{state['symbol']} stock earnings guidance outlook site:bloomberg.com OR site:reuters.com OR site:cnbc.com"

    state["news"] = fetch_news(query, max_hits=5)[: settings.max_news]

    return state


def node_sentiment(state: State) -> State:
    headlines = [f"{n.get('title','')}. {n.get('body','')}" for n in state["news"]]

    state["sentiment"] = sentiment_vader(headlines)

    return state


def node_ta(state: State) -> State:
    df = fetch_bars(state["symbol"], settings.yf_period, settings.yf_interval)
    df = compute_ta(df)

    last = df.iloc[-1].to_dict()
    last["rule_state"] = ta_signal(df.iloc[-1])

    state["ta_df_json"] = json.dumps({"last": last}, default=str)

    return state


def node_fuse(state: State) -> State:
    news_bullets = "\n".join(
        [f"- {n.get('title','')} ({n.get('source','')})" for n in state["news"]]
    )

    query_input = (
        f"TA snapshot JSON:\n{state['ta_df_json']}\n\n"
        f"News summary bullets:\n{news_bullets}\n\n"
        f"Sentiment scores: {state['sentiment']}\n"
    )
    out = llm.invoke([HumanMessage(content=FUSE_PROMPT + "\n\n" + query_input)]).content

    try:
        start = out.find("{")
        end = out.rfind("}")
        state["fuse_json"] = out[start : end + 1]

    except Exception:
        state["fuse_json"] = '{"error":"llm_return_format"}'

    return state


def build_graph() -> StateGraph[State]:
    graph = StateGraph(State)

    graph.add_node("fetch_news", node_fetch_news)
    graph.add_node("sentiment", node_sentiment)
    graph.add_node("ta", node_ta)
    graph.add_node("fuse", node_fuse)

    graph.add_edge(START, "fetch_news")
    graph.add_edge(START, "ta")
    graph.add_edge("fetch_news", "sentiment")
    graph.add_edge("sentiment", "fuse")
    graph.add_edge("ta", "fuse")
    graph.add_edge("fuse", END)

    return graph.compile()
