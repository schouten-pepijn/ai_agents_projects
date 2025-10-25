from typing import Dict, List, TypedDict, Optional
from langchain_core.documents.base import Document


class ResearchState(TypedDict):
    """State for intermediate and final values in the flow."""

    query: str
    sub_questions: List[str]
    retrieved_docs: Dict[str, List[Document]]
    summaries: Dict[str, str]
    answer: Optional[str]
    verification_feedback: Optional[Dict[str, str]]
    refinement_iteration: Optional[int]
    max_iterations: Optional[int]
    failed_questions: Optional[List[str]]
