import json
import logging
from typing import List, Tuple
from langchain_ollama import OllamaChat
from multi_agent_research_v2.config.config import WorkflowConfig, QualityLevel

logger = logging.getLogger("multi_agent_research")


class QualityAssessor:
    """Assess quality of outputs at various stages."""

    def __init__(self, llm: OllamaChat, config: WorkflowConfig):
        self.llm = llm
        self.config = config

    def assess_subquestions(
        self, query: str, sub_questions: List[str]
    ) -> Tuple[float, QualityLevel]:
        """Assess quality of generated sub-questions."""

        if not sub_questions:
            return 0.0, QualityLevel.UNACCEPTABLE

        if len(sub_questions) < self.config.min_subquestions:
            return 0.3, QualityLevel.POOR

        if len(sub_questions) > self.config.max_subquestions:
            return 0.5, QualityLevel.ACCEPTABLE

        prompt_template = f"""Assess the quality of these sub-questions for the main query.
        
Main Query: {query}

Sub-questions:
{json.dumps(sub_questions, indent=2)}

Evaluate on:
1. Relevance to main query (0-10)
2. Coverage of topic (0-10)
3. Non-redundancy (0-10)
4. Clarity (0-10)

Respond with ONLY a JSON object: {{"score": <average 0-10>, "reasoning": "brief explanation"}}"""

        try:
            response = self.llm.invoke(prompt_template)
            result = json.loads(response.strip())
            score = float(result.get("score", 5)) / 10.0

            if score >= 0.85:
                return score, QualityLevel.EXCELLENT

            elif score >= 0.75:
                return score, QualityLevel.GOOD

            elif score >= 0.6:
                return score, QualityLevel.ACCEPTABLE

            elif score >= 0.4:
                return score, QualityLevel.POOR

            else:
                return score, QualityLevel.UNACCEPTABLE

        except Exception as e:
            logger.warning(
                f"Quality assessment failed: {e}. Using default ACCEPTABLE rating."
            )
            return 0.6, QualityLevel.ACCEPTABLE

    def assess_summary(
        self, question: str, summary: str, context: str
    ) -> Tuple[float, QualityLevel]:
        """Assess quality of a summary."""

        if not summary or len(summary) < 20:
            return 0.0, QualityLevel.UNACCEPTABLE

        prompt_template = f"""Assess this summary's quality.

Question: {question}
Summary: {summary}

Rate on:
1. Accuracy (answers the question) (0-10)
2. Completeness (0-10)
3. Conciseness (0-10)
4. Factual grounding (no hallucination) (0-10)

Respond with ONLY a JSON object: {{"score": <average 0-10>, "issues": ["list", "of", "issues"]}}"""

        try:
            response = self.llm.predict(prompt_template)
            result = json.loads(response.strip())
            score = float(result.get("score", 5)) / 10.0

            if score >= 0.85:
                return score, QualityLevel.EXCELLENT

            elif score >= 0.75:
                return score, QualityLevel.GOOD

            elif score >= 0.6:
                return score, QualityLevel.ACCEPTABLE

            elif score >= 0.4:
                return score, QualityLevel.POOR

            else:
                return score, QualityLevel.UNACCEPTABLE

        except Exception as e:
            logger.warning(f"Summary assessment failed: {e}")
            return 0.6, QualityLevel.ACCEPTABLE
