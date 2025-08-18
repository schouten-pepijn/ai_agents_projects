from datetime import datetime
from operator import add
from typing import TypedDict, List, cast, Annotated
from dataclasses import dataclass
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from init_db import init_db

from config import DB_PATH, MODEL_URL, MODEL
from prompts import SYSTEM_PROMPT



init_db()

lc_db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


@dataclass
class ToolCall:
    tstamp: str
    tool: str
    detail: str

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    remaining_steps: int
    
    
def now_iso():
    return datetime.now().isoformat(timespec='seconds') + "Z"


def ensure_select_only(sql: str):
    normalized = " ".join(sql.strip().lower().split())
    
    forbidden = [
        " insert ", " update ", " delete ", " drop ", " alter ", " create ", " replace ", " truncate ", " merge "
    ]
    if normalized.startswith(("insert","update","delete","drop","alter","create","replace","truncate","merge")):
        raise ValueError("Only SELECT queries are allowed.")
    
    for token in forbidden:
        if token in f" {normalized} ":
            raise ValueError("Only SELECT queries are allowed.")
        
        
@tool("list_tables", return_direct=False)
def list_tables_tool() -> str:
    """List available tables in the database."""
    
    tbls = lc_db.get_usable_table_names()
    
    return ", ".join(sorted(tbls)) if tbls else "No tables found."


@tool("describe_tables", return_direct=False)
def describe_tables_tool(tables: str) -> str:
    """Get SQL table info for comma-separated table names."""
    
    names = [t.strip() for t in tables.split(",") if t.strip()]
    if not names:
        return "No valid table names provided."
    
    return lc_db.get_table_info(table_names=names)

@tool("query_sql", return_direct=False)
def query_sql_tool(sql: str) -> str:
    """Run a read-only SQL query on the database."""
    
    ensure_select_only(sql)
    result = lc_db.run(sql)
    
    return str(result)


TOOLS = [list_tables_tool, describe_tables_tool, query_sql_tool]


llm = ChatOllama(
    base_url=MODEL_URL,
    model=MODEL,
    temperature=0.2
)

checkpointer = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    state_schema=AgentState,
    checkpointer=checkpointer,
    prompt=SystemMessage(SYSTEM_PROMPT)
)

def run_agent(user_query: str):
    state: AgentState = {
        "messages": [HumanMessage(user_query)],
        "remaining_steps": 20,    
    }

    # If the user explicitly asks about tables/columns, run schema discovery first and prepend results
    qlow = user_query.lower()
    if any(k in qlow for k in ("table", "tables", "column", "columns", "schema")):
        tbls_list = list(lc_db.get_usable_table_names())
        tbls = ", ".join(sorted(tbls_list)) if tbls_list else "No tables found."
        schema = lc_db.get_table_info(table_names=tbls_list) if tbls_list else "No schema available."
        
        # inject system hint with schema discovery result
        state["messages"].insert(0, SystemMessage(f"Discovered tables: {tbls}\nSchema:\n{schema}"))

    # Stream and capture the final state
    for event in agent.stream(state, {"configurable": {"thread_id": "sql-agent-session"}}, stream_mode="values"):
        state = cast(AgentState, event)

    # Get the final AI message
    final_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    final_text = final_messages[-1].content if final_messages else "(no response)"
    return final_text


if __name__ == "__main__":
    print("DB initialized:", DB_PATH)
    print("Try a few example questions.\n")

    # Example 1
    q1 = "Which customers from NL placed orders, and what is their total amount?"
    ans = run_agent(q1)
    print("Q1:", q1, "\n")
    print(ans, "\n")
    
    print("\n"+"-"*80+"\n")

    # Example 2
    q2 = "Give me the top 3 orders by amount with customer name and status."
    ans = run_agent(q2)
    print("Q2:", q2, "\n")
    print(ans, "\n")
  
    # Example 3 (schema discovery)
    q3 = "Show me the tables and the columns of each."
    ans = run_agent(q3)
    print("Q3:", q3, "\n")
    print(ans, "\n")
   

