from typing import List, TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from config import settings, config
from tools.market import fetch_bars
from tools.news import fetch_news
from tools.sentiment import sentiment_vader
from tools.ta_tools import compute_ta, ta_signal
from prompts import FUSE_PROMPT


class Graphstate(TypedDict):
    symbol: str
    news: List[dict]
    ta_df_json: str
    sentiment: dict
    fuse_json: str
