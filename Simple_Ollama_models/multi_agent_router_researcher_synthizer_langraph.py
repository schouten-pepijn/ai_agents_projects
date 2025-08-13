import os
import re
from typing import TypedDict, Annotated, List, Literal
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

load_dotenv(".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

WEB_SEARCH_TOOL = "duckduckgo"

class GraphState(TypedDict):
    messages: List[BaseMessage]
    notes: str
    sources: List[str]
    route: Literal["search", "synthesize", "final"]
    research_rounds: int       
    last_actor: Literal["router","research","synth"]


router_llm = ChatOllama(
    api_url=OLLAMA_URL,
    model=MODEL,
    temperature=0.0
)

synth_llm = ChatOllama(
    api_url=OLLAMA_URL,
    model=MODEL,
    temperature=0.2
)

research_llm = ChatOllama(
    api_url=OLLAMA_URL,
    model=MODEL,
    temperature=0.1
)

if WEB_SEARCH_TOOL == "tavily":
    web_search = TavilySearch(
        max_results=5,
        tavily_api_key=TAVILY_API_KEY
    )
elif WEB_SEARCH_TOOL == "duckduckgo":
    wrapper = DuckDuckGoSearchAPIWrapper(
        region="wt-wt",
        safesearch="moderate",
        time=None,
        max_results=5
    )
    web_search = DuckDuckGoSearchRun(api_wrapper=wrapper)

tools = [web_search]

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a supervisor that routes a web QA workflow.\n"
     "Read the latest user request and current notes. Decide one of:\n"
     "- 'search'      → if fresh info or citations are needed.\n"
     "- 'synthesize'  → if we already have enough notes to answer.\n"
     "- 'final'       → if we just finished synthesis and can end.\n"
     "Rules:\n"
     "1) If sources are empty and user asks about current events, choose 'search'.\n"
     "2) If the last assistant message came from the Synthesizer, choose 'final'.\n"
     "3) Otherwise, prefer 'synthesize' when notes cover the question.\n"
     "Return ONLY one token: search | synthesize | final."
    ),
    MessagesPlaceholder("messages"),
    ("system", "NOTES (may be empty):\n{notes}\n\nSOURCES (URLs): {sources}")
])

def route_fn(state: GraphState) -> GraphState:
    if state.get("last_actor") == "synth":
        state["route"] = "final"
        return state

    need_fresh = not state.get("sources")  
    if state.get("research_rounds", 0) == 0 and need_fresh:
        state["route"] = "search"
        return state

    MAX_RESEARCH = 2
    if state.get("research_rounds", 0) >= MAX_RESEARCH:
        state["route"] = "synthesize"
        return state

    state["route"] = "synthesize"
    return state


research_prompt = (
    ChatPromptTemplate
    .from_messages([
        ("system",
         "You are a web research specialist.\n"
         "- Use the Tavily search tool when you need fresh info.\n"
         "- Extract crisp bullet-point facts; include direct URLs.\n"
         "- Do NOT write the final answer; only collect notes.\n"
         "- Avoid duplication; if a URL already exists in sources, skip it."
        ),
        MessagesPlaceholder("messages"),
        ("system", "Existing NOTES:\n{notes}\nExisting SOURCES: {sources}\n"
                   "Task: If the user asks for fresh/uncertain info, search and append findings.")
    ])
    .partial(notes="", sources="")
)

research_agent = create_react_agent(
    model=research_llm,
    tools=tools,
    prompt=research_prompt,
    name="Researcher",
)


research_agent = create_react_agent(
    model=research_llm,
    tools=tools,
    prompt=research_prompt,
    name="Researcher",
)


def research_node(state: GraphState) -> GraphState:
    result = research_agent.invoke({
        "messages": state["messages"],
        "notes": state.get("notes", "") or "",
        "sources": ", ".join(state.get("sources", []) or []),
    })

    final_msg: AIMessage = result["messages"][-1]
    new_text = final_msg.content

    urls = re.findall(r'https?://\S+', new_text)
    prev_sources = state.get("sources") or []
    merged_sources = list(dict.fromkeys([*prev_sources, *urls]))
    merged_notes = ((state.get("notes") or "") + "\n" + new_text).strip()

    rounds = state.get("research_rounds", 0) + 1

    return {
        "messages": [*state["messages"], final_msg],
        "notes": merged_notes,
        "sources": merged_sources,
        "route": state["route"],           
        "research_rounds": rounds,
        "last_actor": "research",
    }

synth_prompt = (
    ChatPromptTemplate
    .from_messages([
        ("system",
         "You are a senior analyst. Write a concise, structured answer for the user.\n"
         "Use bullet points, small sections, and include a 'Sources' list of URLs actually present in SOURCES.\n"
         "If information is uncertain, say so explicitly."
        ),
        ("system", "NOTES to use:\n{notes}\n\nSOURCES:\n{sources}"),
        MessagesPlaceholder("messages"),
    ])
    .partial(notes="", sources="")
)

synth_agent = (synth_prompt | synth_llm)


def synth_node(state: GraphState) -> GraphState:
    reply: AIMessage = synth_agent.invoke({
        "notes": state.get("notes", "") or "",
        "sources": "\n".join(f"- {u}" for u in (state.get("sources") or [])),
        "messages": state["messages"],
    })
    return {
        "messages": [*state["messages"], reply],
        "notes": state.get("notes", "") or "",
        "sources": state.get("sources", []) or [],
        "route": "final",               
        "research_rounds": state.get("research_rounds", 0),
        "last_actor": "synth",
    }

    
    
graph = StateGraph(GraphState)

graph.add_node("route", RunnableLambda(route_fn))
graph.add_node("research", RunnableLambda(research_node))
graph.add_node("synthesize", RunnableLambda(synth_node))

graph.set_entry_point("route")


def on_route(state: GraphState) -> str:
    if state["route"] == "search":
        return "research"
    if state["route"] == "synthesize":
        return "synthesize"
    return END

graph.add_conditional_edges("route", on_route)
graph.add_edge("research", "route")
graph.add_edge("synthesize", "route")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer, name="web-multi-agent")



def ask(question: str, thread_id: str = "web-thread-1") -> str:
    result = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "notes": "",
            "sources": [],
            "route": "search",
            "research_rounds": 0,   
            "last_actor": "router",   
        },
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 30
        },
    )
    final_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return final_msgs[-1].content if final_msgs else ""

q = "We have a general ledger account with description 'Cash in Bank'. Determine the right financial category to classify."
print(ask(q))
