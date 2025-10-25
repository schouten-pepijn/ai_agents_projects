from langgraph.graph import StateGraph, START, END
from multi_agent_research_v1.nodes.planner import planner_node
from multi_agent_research_v1.nodes.retrieval import retrieval_node
from multi_agent_research_v1.nodes.summary import summary_node
from multi_agent_research_v1.nodes.verification import verification_node
from multi_agent_research_v1.nodes.synthesis import synthesis_node
from multi_agent_research_v1.core.state import ResearchState


def build_workflow(llm, vector_store) -> StateGraph:
    """Build the research workflow as a sequence of nodes."""

    workflow: StateGraph = StateGraph(state_schema=ResearchState)

    workflow.add_node("planner", lambda state: planner_node(state, llm))
    workflow.add_node("retriever", lambda state: retrieval_node(state, vector_store))
    workflow.add_node("summarizer", lambda state: summary_node(state, llm))
    workflow.add_node("verifier", lambda state: verification_node(state, llm))
    workflow.add_node("synthesizer", lambda state: synthesis_node(state, llm))

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "summarizer")
    workflow.add_edge("summarizer", "verifier")
    workflow.add_edge("verifier", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
