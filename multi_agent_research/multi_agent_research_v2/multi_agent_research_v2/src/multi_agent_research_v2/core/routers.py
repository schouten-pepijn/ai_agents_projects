import logging
from typing import Literal
from multi_agent_research_v2.core.state import ResearchState
from multi_agent_research_v2.config.config import WorkflowConfig

logger = logging.getLogger("multi_agent_research")


def route_after_planner(
    state: ResearchState, config: WorkflowConfig
) -> Literal["retriever", "planner"]:
    """Decide wherher to proceed or refine planning."""
    quality_score = state["quality_scores"].get("planner", 0.0)
    iteration = state["iteration_counts"].get("planner", 0)

    if quality_score >= config.min_quality_score:
        logger.info("Router: Planner quality acceptable, proceeding to retriever.")

        return "retriever"

    if iteration < config.max_refinement_iterations:
        logger.warning(f"Planner quality low ({quality_score:.2f}), refining")

        return "planner"

    logger.warning("Max planner refinements reached, proceeding to retriever.")

    return "retriever"


def route_after_summarizer(
    state: ResearchState, config: WorkflowConfig
) -> Literal["verifier", "summarizer"]:
    """Decide whether summaries need refinement."""
    quality_score = state["quality_scores"].get("summarizer", 0.0)
    iteration = state["iteration_counts"].get("summarizer", 0)

    if quality_score >= config.min_quality_score:
        logger.info("Summarizer quality acceptable, proceeding to verification")

        return "verifier"

    if iteration < config.max_refinement_iterations:
        logger.warning(f"Summarizer quality low ({quality_score:.2f}), refining")

        return "summarizer"

    logger.warning("Summarizer max iterations reached, proceeding anyway")

    return "verifier"


def route_after_verifier(
    state: ResearchState, config: WorkflowConfig
) -> Literal["synthesizer", "summarizer"]:
    """Decide whether to proceed to synthesis or refine summaries."""
    verifier_score = state["quality_scores"].get("verifier", 0.0)
    summarizer_iteration = state["iteration_counts"].get("summarizer", 0)

    if verifier_score >= config.min_quality_score:
        logger.info("Verification passed, proceeding to synthesis")

        return "synthesizer"

    if summarizer_iteration < config.max_refinement_iterations:
        logger.warning("Verification concerns detected, refining summaries")

        return "summarizer"

    logger.warning("Cannot refine further, proceeding to synthesis")

    return "synthesizer"
