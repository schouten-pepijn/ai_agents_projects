from typing import Dict
import logging
from langchain_community.vectorstores.faiss import FAISS
from multi_agent_research_v1.core.state import ResearchState

logger = logging.getLogger("multi_agent_research")


def retrieval_node(
    state: ResearchState, vector_store: FAISS, k: int = 4
) -> ResearchState:
    """Retrieve relevant documents for each sub-question."""
    logger.info("-> RETRIEVER: Fetching relevant documents")

    retrieved = {}
    for question in state.get("sub_questions", []):
        docs_and_scores = vector_store.similarity_search_with_score(question, k=k)
        docs = [doc for doc, score in docs_and_scores]
        retrieved[question] = docs

    state["retrieved_docs"] = retrieved
    logger.info(f"   Retrieved documents for {len(retrieved)} questions")
    return state
