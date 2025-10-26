from typing import TypedDict, List, Dict
from langchain.schema import Document


class ResearchState(TypedDict):

    query: str
    sub_questions: List[str]
    retrieved_docs: Dict[str, List[Document]]
