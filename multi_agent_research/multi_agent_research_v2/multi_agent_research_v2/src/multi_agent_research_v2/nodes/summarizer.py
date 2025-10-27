import logging
from langchain_ollama.chat_models import ChatOllama
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.core.state import ResearchState
from multi_agent_research_v2.nodes.quality_assessor import QualityAssessor

logger = logging.getLogger("multi_agent_research")


class SummarizerNode:
    """Summarization with iterative refinement capability."""

    def __init__(
        self, llm: ChatOllama, assessor: QualityAssessor, config: WorkflowConfig
    ):
        self.llm = llm
        self.assessor = assessor
        self.config = config

    def __call__(self, state: ResearchState) -> ResearchState:
        """Generate summaries with quality assessment."""
        logger.info("Summarizer node: Generating summaries")

        state["current_node"] = "summarizer"

        iteration = state["iteration_counts"].get("summarizer", 0)
        state["iteration_counts"]["summarizer"] = iteration + 1

        summaries = {}
        quality_scores = []

        for question, docs in state.get("retrieved_docs", {}).items():
            if not docs:
                logger.warning(f"No documents for question: {question}")

                summaries[question] = "Insufficient information available."
                quality_scores.append(0.0)

                continue

            context = "\n\n".join(
                [f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
            )

            system_template = """You are a precise research summarization assistant. 
Create factual, concise summaries that directly answer questions using ONLY 
the provided context.

Guidelines:
- Stay grounded in the source material
- Be specific and concrete
- Cite key facts
- Acknowledge limitations if context is insufficient
- Use 2-4 sentences"""

            user_template = f"""Question: {question}

Context:
{context}

Provide a focused summary:"""

            try:
                response = self.llm.invoke(f"{system_template}\n\n{user_template}")
                summary = response.content.strip()

                score, quality = self.assessor.assess_summary(
                    question, summary, context
                )
                quality_scores.append(score)

                summaries[question] = summary

                logger.debug(f"Summary quality for '{question}': {quality.value}")

            except Exception as e:
                logger.error(f"Summarization error: {e}")

                state["errors"].append(f"Summarizer: {str(e)}")

                summaries[question] = "Error generating summary."
                quality_scores.append(0.0)

        state["summaries"] = summaries

        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)

            state["quality_scores"]["summarizer"] = avg_quality

        state["routing_history"].append(f"summarizer:iteration_{iteration}")

        return state
