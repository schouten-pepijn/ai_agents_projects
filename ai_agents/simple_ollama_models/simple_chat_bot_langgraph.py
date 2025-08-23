import os
from typing import TypedDict, Annotated, List
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")
TAVILY_API_KEY = os.getenv("TAVILIY_API_KEY")

llm = ChatOllama(
    api_url=OLLAMA_URL,
    model=MODEL,
    temperature=0.2
)

web_search = TavilySearch(
    max_results=5,
    tavility_api_key=TAVILY_API_KEY
)

tools = [web_search]

SYSTEM_PROMPT = f"""You are a web research agent.
- Use tools when you need FRESH information or citations.
- Prefer concise, structured answers with bullet points and source links.
- If you used web search, ALWAYS list sources.
- Today: {datetime.now().isoformat()} UTC.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("messages"),
])

checkpointer = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
    checkpointer=checkpointer,
    version="v2",
    name="web_research_agent"
)

def run_query(user_prompt: str, thread_id: str = "demo-thread-1"):
    # Each turn we pass a list of messages; prebuilt agent manages the loop.
    result = agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content

q = "Can you find a linkedin link to a Pepijn Schouten from The Netherlands that is a Data Engineer?"

print(run_query(q))