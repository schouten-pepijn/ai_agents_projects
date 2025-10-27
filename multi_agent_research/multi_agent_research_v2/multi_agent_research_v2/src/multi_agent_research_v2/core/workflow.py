import logging
from langgraph.graph import StateGraph, END
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from multi_agent_research_v2.core.routers import (
    route_after_planner,
    route_after_summarizer,
    route_after_verifier,
)
from multi_agent_research_v2.core.state import ResearchState
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.nodes.planner import PlannerNode
from multi_agent_research_v2.nodes.retriever import RetrievalNode
from multi_agent_research_v2.nodes.quality_assessor import QualityAssessor
from multi_agent_research_v2.nodes.summarizer import SummarizerNode
from multi_agent_research_v2.nodes.verifier import VerificationNode
from multi_agent_research_v2.nodes.synthesizer import SynthesisNode

logger = logging.getLogger("multi_agent_research")


def build_workflow(
    llm: ChatOllama, vector_store: FAISS, config: WorkflowConfig
) -> StateGraph:
    """Construct the research workflow graph."""

    assessor = QualityAssessor(llm, config)

    planner = PlannerNode(llm, assessor, config)
    retriever = RetrievalNode(vector_store, config)
    summarizer = SummarizerNode(llm, assessor, config)
    verifier = VerificationNode(llm, config)
    synthesizer = SynthesisNode(llm, config)

    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner)
    workflow.add_node("retriever", retriever)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("verifier", verifier)
    workflow.add_node("synthesizer", synthesizer)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        lambda state: route_after_planner(state, config),
        {"retriever": "retriever", "planner": "planner"},
    )

    workflow.add_edge("retriever", "summarizer")

    workflow.add_conditional_edges(
        "summarizer",
        lambda state: route_after_summarizer(state, config),
        {"verifier": "verifier", "summarizer": "summarizer"},
    )

    workflow.add_conditional_edges(
        "verifier",
        lambda state: route_after_verifier(state, config),
        {"synthesizer": "synthesizer", "verifier": "verifier"},
    )

    workflow.add_edge("synthesizer", END)

    logger.info("Workflow graph constructed successfully.")

    return workflow
