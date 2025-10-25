from typing import Dict, List, TypedDict, Optional
from langchain_core.documents.base import Document


class ResearchState(TypedDict):
    """State for intermediate and final values in the flow."""

    query: str
    sub_questions: List[str]
    retrieved_docs: Dict[str, List[Document]]
    summaries: Dict[str, str]
    answer: Optional[str]
