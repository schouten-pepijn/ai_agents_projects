from typing import Dict, Any
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama.chat_models import ChatOllama
from multi_agent_research_v1.models.schemas import ResearchState, Assessment


def verification_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Verify each summary for completeness and alignment with the sub-question."""

    system_template = (
        "You are a critical reviewer. Evaluate whether the summary below "
        "fully and correctly answers the research sub-question. If the "
        "summary is accurate and complete, respond with 'OK'. Otherwise, "
        "suggest what information is missing or incorrect."
    )

    llm_parsed = llm.with_structured_output(schema=Assessment)

    verified = {}
    for question, summary in state.get("summaries", {}).items():
        user_template = "Sub-question: {question}\nSummary: {summary}\nAssessment:"

        prompt = ChatPromptTemplate.from_messages(
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

        else:
            verified[question] = summary + "\n\n[Reviewer comments: " + assessment + "]"

    state["summaries"] = verified

    return state
