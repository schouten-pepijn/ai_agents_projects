import logging
from langchain_community.vectorstores import FAISS
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.core.state import ResearchState

logger = logging.getLogger("multi_agent_research")


class RetrievalNode:
    """Document retrieval with quality filtering."""

    def __init__(self, vector_store: FAISS, config: WorkflowConfig):
        self.vector_store = vector_store
        self.config = config

    def __call__(self, state: ResearchState) -> ResearchState:
        """Retrieve relevant documents with quality filtering."""
        logger.info("Retrieval node: Fetching relevant documents")

        state["current_node"] = "retriever"

        retrieved = {}
        low_quality_retrievals = []

        for question in state.get("sub_questions", []):
            try:
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    question, k=self.config.retrieval_k
                )

                filtered_docs = [
                    doc
                    for doc, score in docs_and_scores
                    if score >= self.config.similarity_threshold
                ]

                if not filtered_docs:
                    logger.warning(
                        f"No high-quality documents found for question: {question}"
                    )
                    low_quality_retrievals.append(question)

                    filtered_docs = [docs_and_scores[0][0]] if docs_and_scores else []

                retrieved[question] = filtered_docs

                logger.debug(
                    f"Retrieved {len(filtered_docs)} documents for question: {question}"
                )

            except Exception as e:
                logger.error(
                    f"Error retrieving documents for question '{question}': {e}"
                )
                state["errors"].append(
                    f"Retrieval error for question '{question}': {e}"
                )
                retrieved[question] = []

            state["retrieved_docs"] = retrieved

            total_questions = len(state.get("sub_questions", []))
            if total_questions > 0:
                quality_score = 1.0 - (len(low_quality_retrievals) / total_questions)
                state["quality_scores"]["retriever"] = quality_score

            state["routing_history"].append("retriever:complete")

            return state
