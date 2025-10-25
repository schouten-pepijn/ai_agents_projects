from dotenv import load_dotenv
from typing import List
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from multi_agent_research_v1.models.schemas import SubQuestions
from multi_agent_research_v1.core.state import ResearchState

load_dotenv(".env")


def planner_node(state: ResearchState, llm: ChatOllama) -> List[str]:
    """Decompose the query into a set of actionable sub-questions."""

    query = state["query"]

    system_template = (
        "You are a planner assistant tasked with decomposing a research question "
        "into smaller, focused sub-questions. Given the main question below, "
        "generate 2-4 sub-questions that together will comprehensively cover the topic. "
        "Ensure the sub-questions are clear, concise, and unambiguous.\n"
    )

    llm_parsed = llm.with_structured_output(schema=SubQuestions)

    user_template = "Main question: {query}"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
        ]
    )
    messages = prompt.format_messages(query=query)

    response = llm_parsed.invoke(messages)

    state["sub_questions"] = response.sub_questions
    return state
