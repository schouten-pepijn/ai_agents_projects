from graphstate import GraphState
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from nodes import (
    generate_queries,
    web_search,
    synthesize,
    router,
    should_continue
)

def build_graph():
    """
    Constructs and compiles a state graph for managing the flow of operations.
    The graph consists of nodes representing different stages of processing:
    - "generate_queries": Generates queries for further processing.
    - "web_search": Performs web searches based on generated queries.
    - "synthesize": Synthesizes results from the web search.
    - "router": Routes the flow based on conditions.
    Edges define the transitions between nodes:
    - "generate_queries" -> "web_search"
    - "web_search" -> "synthesize"
    - "synthesize" -> "router"
    Conditional edges are added to the "router" node:
    - If `should_continue` evaluates to "loop", the graph loops back to "generate_queries".
    - If `should_continue` evaluates to "final", the graph ends.
    A memory saver is used as a checkpointer to preserve the state during execution.
    Returns:
        StateGraph: A compiled state graph ready for execution.
    """
    builder = StateGraph(GraphState)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("web_search", web_search)
    builder.add_node("synthesize", synthesize)
    builder.add_node("router", router)
    
    builder.set_entry_point("generate_queries")
    builder.add_edge("generate_queries", "web_search")
    builder.add_edge("web_search", "synthesize")
    builder.add_edge("synthesize", "router")
    builder.add_conditional_edges(
        "router",
        should_continue,
        {
            "loop": "generate_queries",
            "final": END
        }
    )
    
    memory = MemorySaver()
    
    return builder.compile(checkpointer=memory)