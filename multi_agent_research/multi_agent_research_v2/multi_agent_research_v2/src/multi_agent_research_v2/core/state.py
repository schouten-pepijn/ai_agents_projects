from typing import TypedDict, List, Dict
from langchain_core.documents import Document


class ResearchState(TypedDict):
    """TypedDict representing the state of the research workflow."""

    query: str
    sub_questions: List[str]
    retrieved_docs: Dict[str, List[Document]]
    summaries: Dict[str, str]
    answer: str | None

    quality_scores: Dict[str, float]
    iteration_counts: Dict[str, int]
    errors: List[str]
    routing_history: List[str]

    status: str
    current_node: str | None


def initialize_research_state(initial_query: str) -> ResearchState:
    """Initialize and return a ResearchState with default values."""

    return ResearchState(
        query=initial_query,
        sub_questions=[],
        retrieved_docs={},
        summaries={},
        answer=None,
        quality_scores={},
        iteration_counts={},
        errors=[],
        routing_history=[],
        status="initialized",
        current_node=None,
    )
