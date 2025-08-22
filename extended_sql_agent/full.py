# pip install -U langgraph langchain langchain_community langchain-ollama duckdb
import os
import re
import time
import json
from typing import Dict, List, Optional, Literal, TypedDict, Any

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# -----------------------------
# 1) Choose LLM (OpenAI or Ollama)
# -----------------------------
# USE_OLLAMA = bool(os.getenv("USE_OLLAMA", "0") == "1")
USE_OLLAMA = True

if USE_OLLAMA:
    # Local model via Ollama (ensure it's running and model pulled, e.g., `ollama pull llama3.1`)
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
                     base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                     temperature=0)
else:
    # OpenAI family via LangChain's init_chat_model shorthand from the tutorial
    from langchain.chat_models import init_chat_model
    llm = init_chat_model(os.getenv("OPENAI_MODEL", "openai:gpt-4.1-mini"), temperature=0)

# -----------------------------
# 2) Database: SQLite or DuckDB
# -----------------------------
USE_DUCKDB = bool(os.getenv("USE_DUCKDB", "0") == "1")

if USE_DUCKDB:
    # Memory DuckDB with a Chinook table copy as a demo
    # You can also `duckdb:///yourfile.duckdb`
    from langchain_community.utilities import SQLDatabase
    import duckdb

    con = duckdb.connect(database=":memory:")
    # Create demo tables quickly (or load your own)
    con.sql("""
    CREATE TABLE Genre(GenreId INTEGER, Name TEXT);
    INSERT INTO Genre VALUES (1,'Rock'),(2,'Jazz'),(3,'Metal'),(4,'Sci Fi & Fantasy');
    CREATE TABLE Track(TrackId INTEGER, Name TEXT, GenreId INTEGER, Milliseconds INTEGER);
    INSERT INTO Track VALUES
        (1,'Song A',1,300000),
        (2,'Song B',1,250000),
        (3,'Song C',4,2900000),
        (4,'Song D',4,2920000),
        (5,'Song E',2,180000);
    """)
    db = SQLDatabase.from_uri("duckdb:///:memory:", engine_args={"creator": lambda: con})
else:
    # SQLite Chinook from tutorial (download or point to local)
    from langchain_community.utilities import SQLDatabase
    import pathlib, requests
    if not pathlib.Path("Chinook.db").exists():
        url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        open("Chinook.db","wb").write(r.content)
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")

DIALECT = db.dialect

# -----------------------------
# 3) Tools
# -----------------------------
from langchain_community.agent_toolkits import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

LIST_TOOL   = next(t for t in tools if t.name == "sql_db_list_tables")
SCHEMA_TOOL = next(t for t in tools if t.name == "sql_db_schema")
QUERY_TOOL  = next(t for t in tools if t.name == "sql_db_query")
CHECK_TOOL  = next(t for t in tools if t.name == "sql_db_query_checker")

list_node   = ToolNode([LIST_TOOL], name="list_tables")
schema_node = ToolNode([SCHEMA_TOOL], name="get_schema")
run_node    = ToolNode([QUERY_TOOL], name="run_query")
# We will invoke CHECK_TOOL explicitly in a node to keep control.

# -----------------------------
# 4) Graph State (extend MessagesState)
# -----------------------------
class SQLState(MessagesState, TypedDict):
    # cached
    table_list: Optional[List[str]]
    table_schemas: Dict[str, str]  # table -> schema text
    # execution
    last_query: Optional[str]
    results_preview: Optional[str]
    error: Optional[str]
    # audit
    audit_log: List[Dict[str, Any]]
    # control
    remaining_steps: int

def init_state(user_question: str, max_steps: int = 12) -> SQLState:
    return {
        "messages": [HumanMessage(content=user_question)],
        "table_list": None,
        "table_schemas": {},
        "last_query": None,
        "results_preview": None,
        "error": None,
        "audit_log": [],
        "remaining_steps": max_steps,
    }

# -----------------------------
# 5) Utility: read-only SQL guard
# -----------------------------
SQL_READ_ONLY_RE = re.compile(r"^\s*(--.*\n\s*)*(SELECT|WITH)\b", re.IGNORECASE | re.DOTALL)

def ensure_select_only(query: str) -> None:
    # Fast path: forbid any DDL/DML keywords; allow SELECT/WITH
    forbidden = re.findall(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE)\b", query, flags=re.I)
    if forbidden:
        raise ValueError(f"Forbidden SQL verb(s): {', '.join(set(forbidden))}")
    if not SQL_READ_ONLY_RE.match(query):
        raise ValueError("Only SELECT/WITH queries are allowed.")

# -----------------------------
# 6) Deterministic list & schema fetch w/ caching
# -----------------------------
def need_tables(state: SQLState) -> bool:
    return state["table_list"] is None

def capture_list_tables(state: SQLState):
    # We will deterministically call the list tables tool (no LLM).
    tool_call = {
        "name": LIST_TOOL.name,
        "args": {},
        "id": "listtables-1",
        "type": "tool_call",
    }
    ai = AIMessage(content="", tool_calls=[tool_call])
    tm = LIST_TOOL.invoke(tool_call)

    # Parse csv-ish names
    tables = [t.strip() for t in tm.content.split(",") if t.strip()]
    return {
        "messages": [ai, tm, AIMessage(content=f"Available tables: {tm.content}")],
        "table_list": tables,
        "audit_log": [{"event": "list_tables", "tables": tables}],
        "remaining_steps": state["remaining_steps"] - 1,
    }

def call_schema_for_relevant(state: SQLState):
    """Force the model to choose relevant tables, but we constrain which tool it may call."""
    # Lightweight relevance prompt (no need to bind all tools)
    sys = (
        "Given the user's request and available table names, pick the MINIMAL set of tables "
        "to inspect. Respond by calling the schema tool with `table_names` as a comma-separated list."
    )
    available = ", ".join(state["table_list"] or [])
    human = f"Question: {state['messages'][0].content}\nAvailable tables: {available}"

    llm_with_tool = llm.bind_tools([SCHEMA_TOOL], tool_choice="any")
    resp = llm_with_tool.invoke([{"role": "system", "content": sys}, {"role": "user", "content": human}])

    return {
        "messages": [resp],
        "audit_log": [{"event": "select_tables_for_schema"}],
        "remaining_steps": state["remaining_steps"] - 1,
    }

def cache_schema(state: SQLState):
    """After schema tool response, cache to state."""
    # The last tool message contains DDL-ish text with sample rows. Cache by table names detected.
    # Simple heuristic: extract names from CREATE TABLE "Name"
    last = state["messages"][-1]
    text = getattr(last, "content", "") or ""
    found = re.findall(r'CREATE TABLE\s+"?([A-Za-z0-9_]+)"?\s*\(', text, flags=re.I)
    tbl_schemas = state["table_schemas"].copy()
    for t in found:
        tbl_schemas[t] = text
    return {
        "table_schemas": tbl_schemas,
        "audit_log": [{"event": "cache_schema", "tables": found}],
    }

# -----------------------------
# 7) Query generation & checking
# -----------------------------
GENERATE_PROMPT = f"""
You are a careful data analyst for a {DIALECT} database.
- Output either a tool call to run the query OR a final natural-language answer if you can answer without SQL.
- Prefer minimal projections, limit to 5 rows UNLESS the user requests otherwise.
- Never select all columns with *.
- Use table and column names exactly as in schema.
- If relevant, show units in the final answer.
"""

def generate_query(state: SQLState):
    sys = {"role": "system", "content": GENERATE_PROMPT}
    # Give the model schemas to condition on (shrunk to last N chars if huge)
    schema_snippet = "\n\n".join(
        f"---\n{t}\n{schema[:2000]}" for t, schema in list(state["table_schemas"].items())
    )
    user_aug = f"{state['messages'][0].content}\n\nSchemas:\n{schema_snippet or '(none yet)'}"
    # Allow model to either answer directly or issue a run_query tool call
    llm_tools = llm.bind_tools([QUERY_TOOL])
    response = llm_tools.invoke([sys, {"role": "user", "content": user_aug}])
    return {
        "messages": [response],
        "audit_log": [{"event": "generate_query"}],
        "remaining_steps": state["remaining_steps"] - 1,
    }

CHECK_PROMPT = f"""
You are a SQL linting assistant for {DIALECT}. Check the query for:
- NOT IN with potential NULLs
- UNION vs UNION ALL
- BETWEEN on exclusive ranges
- Type mismatches, function arity
- Identifier quoting & join keys
- Add LIMIT 5 if absent (unless user asked otherwise)
Return the (possibly rewritten) SQL.
"""

def check_query(state: SQLState):
    # Pull the query from the last tool call args
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls)
    tool_call = last_ai.tool_calls[0]
    candidate = tool_call["args"]["query"]

    # Guard: read-only
    ensure_select_only(candidate)

    # Use the existing checker tool to normalize/fix
    # We'll send the query as a message then force the CHECK_TOOL call.
    checker = llm.bind_tools([CHECK_TOOL], tool_choice="any")
    resp = checker.invoke([{"role": "system", "content": CHECK_PROMPT},
                           {"role": "user", "content": candidate}])

    # Overwrite the tool call id so the ToolNode wiring stays consistent with ReAct lineage
    resp.id = last_ai.id
    return {
        "messages": [resp],
        "last_query": candidate,
        "audit_log": [{"event": "check_query"}],
        "remaining_steps": state["remaining_steps"] - 1,
    }

# -----------------------------
# 8) Retry wrapper for run_query (db errors)
# -----------------------------
def should_retry_db_error(msg: str) -> bool:
    # Add patterns you want to auto-retry on
    return any(pat in msg.lower() for pat in [
        "deadlock", "timeout", "locked", "try again", "transient"
    ])

def backoff_sleep(i: int):
    time.sleep(min(0.25 * (2 ** i), 3.0))

def run_query_with_retry(state: SQLState):
    # Defer actual execution to ToolNode but handle post-processing here by reading ToolMessage
    # We simply chain to run_node by returning; the ToolNode will produce a ToolMessage.
    return {}

def capture_results(state: SQLState):
    """Read the last ToolMessage content, stash a compact preview, and optionally retry if error looked transient."""
    last = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    err = None
    if "Traceback" in content or "Error" in content:
        # heuristic; your SQL toolkit returns clean tuples typically.
        if should_retry_db_error(content) and state["remaining_steps"] > 0:
            backoff_sleep(1)
            return {
                "audit_log": [{"event": "db_retry", "message": content}],
                "remaining_steps": state["remaining_steps"] - 1,
            }
        err = content

    # results often like: [('Sci Fi & Fantasy', 2911783.0384)] from toolkit
    preview = content
    if len(content) > 2000:
        preview = content[:2000] + "…"

    return {
        "results_preview": preview,
        "error": err,
        "audit_log": [{"event": "results", "preview": preview[:200]}],
    }

# -----------------------------
# 9) Summarize final answer
# -----------------------------
FINAL_PROMPT = """
Summarize the SQL result for a technical user:
- Be precise and concise.
- Include a compact table in Markdown if rows/columns are present.
- State assumptions or filters if any.
- If units or time ranges are implicit, make them explicit.
"""

def finalize(state: SQLState):
    if state["error"]:
        # Surface failure crisply
        return {
            "messages": [AIMessage(content=f"Query failed safely (read-only mode preserved).\n\nDetails:\n{state['error']}")]
        }

    # Build a compact “data note” the LLM can use to render a table
    data_note = state["results_preview"] or "(no results)"
    prompt = [
        {"role": "system", "content": FINAL_PROMPT},
        {"role": "user", "content": f"User question: {state['messages'][0].content}\n\nRaw DB result:\n{data_note}"}
    ]
    resp = llm.invoke(prompt)
    return {"messages": [resp]}

# -----------------------------
# 10) Wiring the graph
# -----------------------------
def has_tool_call(state: SQLState) -> Literal["check", "finalize"]:
    last = state["messages"][-1]
    return "check" if isinstance(last, AIMessage) and last.tool_calls else "finalize"

def continue_or_stop(state: SQLState) -> Literal["run", END]:
    # If last message has a CHECK_TOOL tool-call, we need to execute it -> then run_query
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "run"
    return END

graph = StateGraph(SQLState)

# Nodes
graph.add_node("list_tables_det", capture_list_tables)
graph.add_node("select_schema", call_schema_for_relevant)
graph.add_node("cache_schema", cache_schema)
graph.add_node("get_schema", schema_node)            # tool
graph.add_node("generate_query", generate_query)
graph.add_node("check_query", check_query)
graph.add_node("run_query", run_node)                # tool
graph.add_node("capture_results", capture_results)
graph.add_node("finalize", finalize)

# Edges
graph.add_edge(START, "list_tables_det")
graph.add_edge("list_tables_det", "select_schema")
graph.add_edge("select_schema", "get_schema")
graph.add_edge("get_schema", "cache_schema")
graph.add_edge("cache_schema", "generate_query")
graph.add_conditional_edges("generate_query", has_tool_call, {"check": "check_query", "finalize": "finalize"})
graph.add_edge("check_query", "run_query")
graph.add_edge("run_query", "capture_results")
graph.add_edge("capture_results", "generate_query")  # allows iterative refinement (LLM can adjust query)
# If generate_query eventually returns no tool call, we end at finalize
graph.add_edge("finalize", END)

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

# -----------------------------
# 11) Run
# -----------------------------
if __name__ == "__main__":
    q = "Which genre on average has the longest tracks?"
    state = init_state(q, max_steps=12)

    # stream values and print just the last message incrementally
    for event in agent.stream(state, stream_mode="values"):
        msg = event["messages"][-1]
        role = msg.__class__.__name__.replace("Message","")
        print(f"\n[{role}]")
        if getattr(msg, "tool_calls", None):
            print("Tool Calls:", msg.tool_calls)
        else:
            print((msg.content[:500] + "…") if len(msg.content) > 500 else msg.content)
