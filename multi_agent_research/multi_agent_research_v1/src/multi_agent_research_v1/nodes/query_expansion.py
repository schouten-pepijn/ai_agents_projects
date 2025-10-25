from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from multi_agent_research_v1.models.schemas import SubQuestions
from multi_agent_research_v1.core.state import ResearchState


def query_expansion_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Expand sub-questions that didn't retrieve sufficient information."""

    feedback = state.get("verification_feedback", {})

    system_template = (
        "You are a query reformulation expert. Given a sub-question that didn't "
        "yield good results and reviewer feedback, generate 1-2 alternative "
        "formulations that might retrieve better information."
    )

    llm_parsed = llm.with_structured_output(schema=SubQuestions)
    expanded_questions = []

    for question, fb in feedback.items():
        if fb != "OK":
            user_template = "Original question: {question}\nFeedback: {feedback}\nAlternative questions:"

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template(user_template),
                ]
            )
            messages = prompt.format_messages(question=question, feedback=fb)

            response = llm_parsed.invoke(messages)
            expanded_questions.extend(response.sub_questions)

    if expanded_questions:
        state["sub_questions"].extend(expanded_questions)

    return state
