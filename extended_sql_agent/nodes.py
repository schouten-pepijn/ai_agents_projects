
from extended_sql_agent.states import SQLState
from tools import LIST_TOOL

def need_tables(state: SQLState) -> bool:
    return state["table_list"] is None

def capture_list_tables(state: SQLState):
    
    tool_call {
        "name": LIST_TOOL.name,
        "args": {},
        "id": "listtables-1",
        "type": "tool_call"
    }