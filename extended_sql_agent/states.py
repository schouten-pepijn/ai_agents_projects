
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Optional, List, Dict, Any, Annotated, TypedDict


class SQLState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
    table_list: Optional[list[str]]
    table_schema: Dict[str, str]
    last_query: Optional[str]
    results_preview: Optional[str]
    error: Optional[str]
    audit_log: List[Dict[str, Any]]
    remaining_steps: int


def init_state(user_question: str, max_steps: int = 12) -> SQLState:
    return SQLState(
        messages=[HumanMessage(content=user_question)],
        table_list=None,
        table_schema={},
        last_query=None,
        results_preview=None,
        error=None,
        audit_log=[],
        remaining_steps=max_steps
    )