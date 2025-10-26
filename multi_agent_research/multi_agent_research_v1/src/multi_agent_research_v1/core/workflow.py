from langgraph.graph import StateGraph, START, END
import logging
from multi_agent_research_v1.nodes.planner import planner_node
from multi_agent_research_v1.nodes.retrieval import retrieval_node
from multi_agent_research_v1.nodes.summary import summary_node
from multi_agent_research_v1.nodes.verification import verification_node
from multi_agent_research_v1.nodes.synthesis import synthesis_node
from multi_agent_research_v1.nodes.query_expansion import query_expansion_node
from multi_agent_research_v1.core.state import ResearchState

logger = logging.getLogger("multi_agent_research")


def build_workflow(llm, vector_store) -> StateGraph:
    """Build the research workflow as a sequence of nodes."""

    workflow: StateGraph = StateGraph(state_schema=ResearchState)

    workflow.add_node("planner", lambda state: planner_node(state, llm))
    workflow.add_node("retriever", lambda state: retrieval_node(state, vector_store))
    workflow.add_node("summarizer", lambda state: summary_node(state, llm))
    workflow.add_node("verifier", lambda state: verification_node(state, llm))
    workflow.add_node("query_expander", lambda state: query_expansion_node(state, llm))
    workflow.add_node("synthesizer", lambda state: synthesis_node(state, llm))

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "summarizer")
    workflow.add_edge("summarizer", "verifier")

    workflow.add_conditional_edges(
        "verifier",
        needs_refinement,
        {"query_expander": "query_expander", "synthesizer": "synthesizer"},
    )
    workflow.add_edge("query_expander", "retriever")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


def needs_refinement(state: ResearchState) -> str:
    """Determine if any summaries need refinement based on verification results."""
    iteration = state.get("refinement_iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if iteration >= max_iter:
        logger.info(f"-> ROUTING: Max iterations ({max_iter}) reached → SYNTHESIZER")
        return "synthesizer"

    feedback = state.get("verification_feedback", {})
    for question, fb in feedback.items():
        if fb != "OK":
            logger.info(
                f"-> ROUTING: Issues found (iteration {iteration}/{max_iter}) → QUERY_EXPANDER"
            )
            return "query_expander"

    logger.info("-> ROUTING: All summaries verified → SYNTHESIZER")
    return "synthesizer"
