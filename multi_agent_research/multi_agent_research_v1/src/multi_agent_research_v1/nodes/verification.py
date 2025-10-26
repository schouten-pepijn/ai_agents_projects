from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama.chat_models import ChatOllama
from multi_agent_research_v1.models.schemas import Assessment
from multi_agent_research_v1.core.state import ResearchState


def verification_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Verify each summary for completeness and alignment with the sub-question."""

    system_template = (
        "You are a critical reviewer. Evaluate whether the summary below "
        "fully and correctly answers the research sub-question. If the "
        "summary is accurate and complete, respond with 'OK'. Otherwise, "
        "suggest what information is missing or incorrect. Give a confidence "
        "level between 0.0 and 1.0."
    )

    llm_parsed = llm.with_structured_output(schema=Assessment)

    verified = {}
    feedback = state.get("verification_feedback", {})

    for question, summary in state.get("summaries", {}).items():
        previous_feedback = feedback.get(question, "")

        context = (
            f"Previous feedback: {previous_feedback}\n" if previous_feedback else ""
        )

        user_template = (
            context + "Sub-question: {question}\nSummary: {summary}\nAssessment:"
        )

        prompt = ChatPromptTemplate(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template),
            ]
        )
        messages = prompt.format_messages(question=question, summary=summary)

        response = llm_parsed.invoke(messages)

        assessment = response.assessment.strip()
        if assessment.upper() == "OK":
            verified[question] = summary
            feedback[question] = "OK"

        else:
            verified[question] = summary
            feedback[question] = assessment

    state["summaries"] = verified
    state["verification_feedback"] = feedback

    return state
