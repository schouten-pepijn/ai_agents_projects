from langchain_ollama.chat_models import ChatOllama
import logging
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from multi_agent_research_v1.models.schemas import SubQuestions
from multi_agent_research_v1.core.state import ResearchState

logger = logging.getLogger("multi_agent_research")


def query_expansion_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Replace only failed sub-questions with better alternatives."""

    iteration = state.get("refinement_iteration", 0)
    state["refinement_iteration"] = iteration + 1

    logger.info(
        f"-> QUERY_EXPANDER: Reformulating failed questions (iteration {state['refinement_iteration']})"
    )

    feedback = state.get("verification_feedback", {})

    passed_questions = {q for q, fb in feedback.items() if fb == "OK"}
    failed_questions = {q for q, fb in feedback.items() if fb != "OK"}

    if not failed_questions:
        logger.info("   No questions need reformulation")
        return state

    logger.info(
        f"   Keeping {len(passed_questions)} successful, reformulating {len(failed_questions)} failed"
    )

    system_template = (
        "You are a query reformulation expert. Given a sub-question that didn't "
        "yield good results and reviewer feedback, generate ONE better alternative."
    )

    llm_parsed = llm.with_structured_output(schema=SubQuestions)

    new_questions = []
    for question in failed_questions:
        fb = feedback[question]

        user_template = (
            "Original question: {question}\n"
            "Feedback: {feedback}\n"
            "Generate ONE improved question:"
        )

        prompt = ChatPromptTemplate(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template),
            ]
        )
        messages = prompt.format_messages(question=question, feedback=fb)

        response = llm_parsed.invoke(messages)
        if response.sub_questions:
            new_questions.append(response.sub_questions[0])

    state["sub_questions"] = list(passed_questions) + new_questions

    for question in failed_questions:
        state["retrieved_docs"].pop(question, None)
        state["summaries"].pop(question, None)
        state["verification_feedback"].pop(question, None)

    logger.info(
        f"   New question set: {len(state['sub_questions'])} total ({len(passed_questions)} kept + {len(new_questions)} new)"
    )

    return state
