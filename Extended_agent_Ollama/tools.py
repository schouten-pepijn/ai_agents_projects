from typing import List
from langchain.tools import Tool
from memory_vectorstore import LTMemory
from datetime import datetime

ltm = LTMemory()

ltm.add_texts([
    "Paris is the capital of France",
     "The Pacific Ocean is the largest ocean on Earth",
     "A good incident runbook has detect, diagnose, mitigate and verify steps"
])

def calc(expr: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expr) <= allowed:
        raise ValueError("Invalid characters in expression")
    
    return str(eval(expr))

def now_iso() -> str:
    return datetime.now(datetime.timezone.utc).isoformat()

def kb_search(query: str) -> str:
    docs = ltm.retriever(k=3).invoke(query)
    return "\n\n".join(f"- {d.page_content}" for d in docs)

TOOLS: List[Tool] = [
    Tool.from_function(
        calc,
        name="calculator",
        description="Calculate a mathematical expression"
    ),
    Tool.from_function(
        now_iso,
        name="utc_time_now",
        description="Get the current time in ISO format"
    ),
    Tool.from_function(
        kb_search,
        name="knowledge_base_search",
        description="Search the long-term knowledge base"
    ),
]