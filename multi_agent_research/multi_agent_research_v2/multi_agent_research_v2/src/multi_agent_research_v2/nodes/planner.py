import logging
from langchain_ollama.chat_models import ChatOllama
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.core.state import ResearchState
from multi_agent_research_v2.nodes.quality_assessor import QualityAssessor
from multi_agent_research_v2.nodes.schemas import SubQuestionsOutput

logger = logging.getLogger("multi_agent_research")


class PlannerNode:
    """Planning node with quality validation and refinement."""

    def __init__(
        self, llm: ChatOllama, assessor: QualityAssessor, config: WorkflowConfig
    ):
        self.llm = llm
        self.config = config
        self.assessor = assessor

    def __call__(self, state: ResearchState) -> ResearchState:
        """Generate and validate sub-questions."""
        logger.info("Planner node: Decomposing query")

        state["current_node"] = "planner"

        iteration = state["iteration_counts"].get("planner", 0)
        state["iteration_counts"]["planner"] = iteration + 1

        if iteration >= self.config.max_refinement_iterations:
            logger.warning("Max planner iterations reached")
            state["errors"].append("Planner: Max iterations exceeded")

            return state

        query = state["query"]

        prompt = f"""You are an expert research planning assistant. Decompose the following complex 
research question into focused, actionable sub-questions.

Guidelines:
- Generate 2-5 sub-questions
- Each should be specific and answerable
- Cover different aspects of the main query
- Avoid redundancy
- Ensure logical flow

Main research query: {query}"""

        try:
            structured_llm = self.llm.with_structured_output(SubQuestionsOutput)
            result = structured_llm.invoke(prompt)

            sub_questions = result.sub_questions

            score, quality = self.assessor.assess_subquestions(query, sub_questions)
            state["quality_scores"]["planner"] = score

            logger.info(f"Planner quality: {quality.value} (score: {score:.2f})")

            state["sub_questions"] = sub_questions
            state["routing_history"].append(
                f"planner:iteration_{iteration}_quality_{quality.value}"
            )

        except Exception as e:
            logger.error(f"Planner node error: {e}")
            state["errors"].append(f"Planner error: {str(e)}")

            state["sub_questions"] = [
                f"What are the key concepts related to: {query}?",
                f"What are the main challenges to consider for: {query}?",
            ]
            state["quality_scores"]["planner"] = 0.5

        return state
