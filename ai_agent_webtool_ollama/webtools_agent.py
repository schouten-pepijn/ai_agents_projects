import os
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler



load_dotenv(".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL_EXT")  # Changed to getenv for safety

WEB_SEARCH_TOOL = "duckduckgo"

@dataclass
class ToolCall:
    name: str
    input: str
    output: str


class ToolCollector(BaseCallbackHandler):
    """
    Callback handler to collect tool calls made by the agent.
    """
    def __init__(self):
        self.calls = []

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any):
        tool_name = serialized.get("name", "unknown_tool")
        self.calls.append(ToolCall(name=tool_name, input=input_str, output=""))

    def on_tool_end(self, output: Any, **kwargs: Any):
        for i in range(len(self.calls) -1, -1, -1):
            if self.calls[i].output is None:
                self.calls[i].output = output
                break

    def on_tool_error(self, error: Exception, **kwargs: Any):
        self.calls.append(ToolCall(name="tool_error", input="", output=str(error)))
        


def build_agent():
    """
    Build and return a LangChain agent with DuckDuckGo search tool.
    """
    load_dotenv()
    base_url = os.getenv("BASE_URL")
    model_id = os.getenv("MODEL_EXT")

    if not base_url or not model_id:
        raise ValueError("BASE_URL and MODEL must be set in environment variables.")
    
    llm = ChatOllama(
        base_url=base_url,
        model=model_id,
        temperature=0.2
    )
    
    ddg_results = DuckDuckGoSearchResults(name="duckduckgo_search")
    tools = [ddg_results]
    llm_agent = create_react_agent(llm, tools=tools)
    
    return llm_agent

def extract_sources_from_output(output: Any) -> List[Tuple[str, Optional[str]]]:
    sources = []
    
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                url = item.get("link") or item.get("url")
                title = item.get("title")
                if url:
                    sources.append((url, title))

    return sources


def print_tool_calls_and_sources(calls: List[ToolCall]) -> None:
    print("\n=== TOOL CALLS ===")
    for i, tc in enumerate(calls, 1):
        print(f"{i}. tool={tc.name}")
        if tc.input:
            trimmed = (tc.input[:500] + "…") if len(tc.input) > 500 else tc.input
            print(f"   input: {trimmed!r}")

        if isinstance(tc.output, (str, bytes)):
            preview = tc.output[:300] + ("…" if len(tc.output) > 300 else "")
            print(f"   output (preview): {preview!r}")
        else:
            print(f"   output (type): {type(tc.output).__name__}")

    all_sources: List[Tuple[str, Optional[str]]] = []
    for tc in calls:
        all_sources.extend(extract_sources_from_output(tc.output))

    seen = set()
    unique_sources = []
    for url, title in all_sources:
        if url not in seen:
            unique_sources.append((url, title))
            seen.add(url)

    if unique_sources:
        print("\n=== WEB SOURCES ===")
        for j, (url, title) in enumerate(unique_sources, 1):
            if title:
                print(f"{j}. {title} — {url}")
            else:
                print(f"{j}. {url}")
    else:
        print("\n=== WEB SOURCES ===")
        print("No structured sources found (did the agent use the *results* tool?).")


def run(query: str, stream: bool = False) -> None:
    agent = build_agent()
    collector = ToolCollector()

    if stream:
        # Stream the reasoning/messages AND collect tool calls via callbacks
        print("\n--- Agent run (streaming) ---\n")
        for event in agent.stream(
            {"messages": [HumanMessage(query)]},
            stream_mode="values",
            config={"callbacks": [collector], "run_name": "websearch_run"},
        ):
            msgs = event.get("messages", [])
            if msgs:
                msg = msgs[-1]
                role = type(msg).__name__.replace("Message", "")
                print(f"[{role}] {getattr(msg, 'content', '')}")
        print("\n--- End ---")
    else:
        # Single-shot invoke
        result = agent.invoke(
            {"messages": [HumanMessage(query)]},
            config={"callbacks": [collector], "run_name": "websearch_run"},
        )
        print("\n--- Agent answer ---\n")
        print(result["messages"][-1].content)

    # Always print tools and sources at the end
    print_tool_calls_and_sources(collector.calls)


if __name__ == "__main__":
    run("Summarize what a general ledger is. Use a tool and keep it short.", stream=True)