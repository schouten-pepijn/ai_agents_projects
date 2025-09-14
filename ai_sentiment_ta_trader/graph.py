import json

from typing import List, TypedDict, Literal, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import config, settings
from market import fetch_bars
from news import fetch_news
from sentiment import sentiment_vader
from ta_tools import compute_ta, ta_signal
from prompts import FUSE_PROMPT


def keep_first(x, y):
    return x if x else y


class State(TypedDict):
    symbol: Annotated[str, keep_first]
    news: Annotated[List[dict], add]
    ta_df_json: Annotated[str, keep_first]
    sentiment: Annotated[dict, keep_first]
    fuse_json: Annotated[str, keep_first]


llm = ChatOllama(
    model=config.ollama_model, base_url=config.ollama_base_url, temperature=0.1
)


def node_fetch_news(state: State) -> State:
    query = f"{state['symbol']} stock earnings guidance outlook"

    state["news"] = fetch_news(query, max_hits=5)[: settings.max_news]

    return state


def node_sentiment(state: State) -> State:
    headlines = [f"{n.get('title','')}. {n.get('body','')}" for n in state["news"]]

    state["sentiment"] = sentiment_vader(headlines)

    return state


def node_ta(state: State) -> State:
    print("\n=== NODE_TA DEBUG ===")
    print(f"Fetching data for symbol: {state['symbol']}")

    df = fetch_bars(state["symbol"], settings.yf_period, settings.yf_interval)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")

    df = compute_ta(df)
    print(f"DataFrame shape after TA: {df.shape}")
    print(f"DataFrame columns after TA: {df.columns.tolist()}")

    last = df.iloc[-1].to_dict()
    print(f"Last row keys: {list(last.keys())}")

    last["rule_state"] = ta_signal(df.iloc[-1])
    print(f"Rule state: {last['rule_state']}")

    state["ta_df_json"] = json.dumps({"last": last}, default=str)
    print(f"TA JSON length: {len(state['ta_df_json'])}")
    print(f"TA JSON preview: {state['ta_df_json'][:200]}...")

    return state


def node_fuse(state: State) -> State:
    print("\n=== NODE_FUSE DEBUG ===")
    print(f"TA data length: {len(state['ta_df_json'])}")
    print(f"TA data: {state['ta_df_json']}")
    print(f"News count: {len(state['news'])}")
    print(f"Sentiment: {state['sentiment']}")

    news_bullets = "\n".join(
        [f"- {n.get('title','')} ({n.get('source','')})" for n in state["news"]]
    )

    query_input = (
        f"TA snapshot JSON:\n{state['ta_df_json']}\n\n"
        f"News summary bullets:\n{news_bullets}\n\n"
        f"Sentiment scores: {state['sentiment']}\n"
    )

    print(f"Query input length: {len(query_input)}")
    print("Calling LLM...")

    try:
        out = llm.invoke(
            [HumanMessage(content=FUSE_PROMPT + "\n\n" + query_input)]
        ).content
        print(f"LLM response length: {len(out)}")
        print(f"LLM response: {out}")

        start = out.find("{")
        end = out.rfind("}")

        print(f"JSON start index: {start}, end index: {end}")

        if start != -1 and end != -1 and end > start:
            state["fuse_json"] = out[start : end + 1]
            print(f"Extracted JSON: {state['fuse_json']}")
        else:
            state["fuse_json"] = '{"error":"no_json_found"}'
            print("No valid JSON found in LLM response")

    except Exception as e:
        print(f"Exception in node_fuse: {e}")
        state["fuse_json"] = f'{{"error":"llm_exception","message":"{str(e)}"}}'

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
