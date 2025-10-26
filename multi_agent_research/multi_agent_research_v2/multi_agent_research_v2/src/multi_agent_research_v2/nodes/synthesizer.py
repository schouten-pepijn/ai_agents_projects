import logging
import json
from langchain_ollama import ChatOllama
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.core.state import ResearchState

logger = logging.getLogger("multi_agent_research")


class SynthesisNode:
    """Final synthesis with quality validation."""

    def __init__(self, llm: ChatOllama, config: WorkflowConfig):
        self.llm = llm
        self.config = config

    def __call__(self, state: ResearchState) -> ResearchState:
        """Synthesize comprehensive final answer."""
        logger.info("Synthesis node: Creating final answer")

        state["current_node"] = "synthesizer"

        iteration = state["iteration_counts"].get("synthesizer", 0)
        state["iteration_counts"]["synthesizer"] = iteration + 1

        query = state["query"]
        summaries = state.get("summaries", {})

        if not summaries:
            state["answer"] = "Unable to generate answer: no summaries available."
            state["quality_scores"]["synthesizer"] = 0.0

            return state

        combined = "\n\n".join(
            [f"Sub-question: {q}\nFindings: {s}" for q, s in summaries.items()]
        )

        system_template = """You are an expert research synthesis assistant. Create a 
comprehensive, well-structured answer that:

1. Directly addresses the main research question
2. Integrates findings from all sub-questions
3. Maintains logical flow and coherence
4. Cites specific information where relevant
5. Acknowledges any limitations or gaps
6. Uses clear, professional language

Structure: Introduction → Main findings → Synthesis → Conclusion"""

        user_template = f"""Main Research Question: {query}

Research Findings:
{combined}

Synthesize a comprehensive answer:"""

        try:
            response = self.llm.invoke(f"{system_template}\n\n{user_template}")
            answer = response.strip()

            answer_length = len(answer)
            if answer_length < self.config.min_answer_length:
                logger.warning(f"Answer too short: {answer_length} chars")

                state["quality_scores"]["synthesizer"] = 0.4

            elif answer_length > self.config.max_answer_length:
                logger.warning(f"Answer too long: {answer_length} chars")

                state["quality_scores"]["synthesizer"] = 0.7

            else:
                state["quality_scores"]["synthesizer"] = 0.9

            state["answer"] = answer

            logger.info("Synthesis complete")

        except Exception as e:
            logger.error(f"Synthesis error: {e}")

            state["errors"].append(f"Synthesizer: {str(e)}")
            state["answer"] = "Error generating final answer."
            state["quality_scores"]["synthesizer"] = 0.0

        state["routing_history"].append(f"synthesizer:iteration_{iteration}")
        state["status"] = "complete"

        return state
