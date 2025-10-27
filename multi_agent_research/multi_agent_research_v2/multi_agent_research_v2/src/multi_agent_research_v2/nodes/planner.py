import logging
import json
import re
from langchain_ollama.chat_models import ChatOllama
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.core.state import ResearchState
from multi_agent_research_v2.nodes.quality_assessor import QualityAssessor

logger = logging.getLogger("multi_agent_research")


def extract_json(text: str) -> str:
    """Extract JSON from text that might contain markdown or other formatting."""
    text = text.strip()

    # Try to find JSON object in the text
    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Try to find JSON array in the text
    array_match = re.search(r"\[[^\]]+\]", text, re.DOTALL)
    if array_match:
        return array_match.group(0)

    return text


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

        system_template = """You are an expert research planning assistant. Decompose complex 
research questions into focused, actionable sub-questions.

Guidelines:
- Generate 2-5 sub-questions
- Each should be specific and answerable
- Cover different aspects of the main query
- Avoid redundancy
- Ensure logical flow

Respond with ONLY a JSON array of strings: ["question 1", "question 2", ...]"""

        user_template = f"Main research query: {query}\n\nGenerate sub-questions:"

        try:
            response = self.llm.invoke(f"{system_template}\n\n{user_template}")

            content = response.content.strip()
            json_str = extract_json(content)
            sub_questions = json.loads(json_str)

            if not isinstance(sub_questions, list):
                raise ValueError("Response is not a list")

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
