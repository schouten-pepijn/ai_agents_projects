from graphstate import GraphState
from langchain_core.messages import HumanMessage
from typing import List, Dict


def _get_user_question(state: GraphState) -> str:
    """
    Extracts the most recent user question from the conversation state.
    Iterates through the messages in reverse order to find the latest instance of a HumanMessage.
    Returns the content of that message. If no HumanMessage is found, returns the content of the first message,
    or an empty string if there are no messages.
    Args:
        state (GraphState): The current conversation state containing messages.
    Returns:
        str: The content of the most recent user question, or an empty string if none exists.
    """
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    
    return state["messages"][0].content if state["messages"] else ""


def _compact_sources_for_llm(hits: List[Dict], max_chars: int = 8000) -> str:
    """
    Compacts a list of source hits into a formatted string suitable for LLM input.
    Args:
        hits (List[Dict]): A list of dictionaries, each representing a source with keys "title", "url", and "content".
        max_chars (int, optional): Maximum number of characters to include in the output string. Defaults to 8000.
    Returns:
        str: A string containing up to 50 formatted source entries, each including the title, URL, and a snippet of content, truncated to fit within max_chars.
    """
    lines = []
    for i, h in enumerate(hits[:50], start=1):
        title = (h.get("title") or "").strip()
        url = (h.get("url") or "").strip()
        content = (h.get("content") or "").replace("\n", "")
        snippet = content[:500]
        lines.append(f"[{i}] {title} | {url} :: {snippet}")
    
    blob = "\n".join(lines)
    
    return blob[:max_chars]
    
    
def _sources_table(hits: List[Dict]) -> List[Dict]:
    """
    Converts a list of source dictionaries into a table format.
    Each source dictionary in `hits` should contain at least 'title' and 'url' keys.
    The function creates a list of rows, where each row is a list containing:
        - an incremental ID (starting from 1),
        - the source title (truncated to 120 characters),
        - the source URL.
    Args:
        hits (List[Dict]): A list of dictionaries representing sources, each with 'title' and 'url' keys.
    Returns:
        List[List]: A list of rows, where each row is a list [id, title, url].
    """
    table = []
    for i, h in enumerate(hits, start=1):
        table.append({
            "id": i,
            "title": h.get("title", "")[:120],
            "url": h.get("url", ""),
        })
        
    rows = [[s.get("id"), s.get("title"), s.get("url")] for s in table]

    return rows