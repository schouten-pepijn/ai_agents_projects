from typing import TypedDict, List, Dict
from langchain_core.documents import Document


class ResearchState(TypedDict):

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
