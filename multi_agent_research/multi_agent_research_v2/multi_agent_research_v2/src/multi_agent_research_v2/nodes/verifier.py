import logging
from langchain_ollama import ChatOllama
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.core.state import ResearchState
from multi_agent_research_v2.nodes.schemas import VerificationOutput

logger = logging.getLogger("multi_agent_research")


class VerificationNode:
    """Enhanced verification with detailed feedback."""

    def __init__(self, llm: ChatOllama, config: WorkflowConfig):
        self.llm = llm
        self.config = config

    def __call__(self, state: ResearchState) -> ResearchState:
        """Verify summaries for accuracy and completeness."""
        logger.info("Verification node: Checking summaries")

        state["current_node"] = "verifier"

        verified = {}
        verification_scores = []

        for question, summary in state.get("summaries", {}).items():
            prompt = f"""You are a critical research verifier. Evaluate the following summary for:
1. Factual accuracy
2. Completeness (answers the question)
3. Clarity
4. Absence of hallucination

Question: {question}
Summary: {summary}

Provide your evaluation with status ('pass' or 'fail'), score (0-10), and detailed feedback."""

            try:
                structured_llm = self.llm.with_structured_output(VerificationOutput)
                result = structured_llm.invoke(prompt)

                status = result.status
                score = result.score / 10.0
                feedback = result.feedback

                verification_scores.append(score)

                if status == "pass" and score >= self.config.min_quality_score:
                    verified[question] = summary
                    logger.debug(f"Verification passed for: {question}")

                else:
                    verified[question] = f"{summary}\n\n[Verification: {feedback}]"
                    logger.warning(f"Verification concern for '{question}': {feedback}")

            except Exception as e:
                logger.error(f"Verification error: {e}")

                verified[question] = summary
                verification_scores.append(0.5)

        state["summaries"] = verified

        if verification_scores:
            state["quality_scores"]["verifier"] = sum(verification_scores) / len(
                verification_scores
            )

        state["routing_history"].append("verifier:complete")

        return state
