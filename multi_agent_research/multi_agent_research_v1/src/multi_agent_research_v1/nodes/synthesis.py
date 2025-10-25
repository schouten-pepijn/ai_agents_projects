from typing import Dict, List
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.documents.base import Document
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field
from multi_agent_research_v1.models.schemas import ResearchState, FinalAnswer


def synthesis_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Synthesize the verified summaries into a final answer."""
    query = state["query"]

    combined_summaries = "\n\n".join(
        [
            f"Sub-question: {question}\nSummary: {summary}"
            for question, summary in state.get("summaries", {}).items()
        ]
    )

    system_template = (
        "You are a research synthesis agent. Given a user query and verified "
        "summaries of research sub-questions, write a comprehensive, well-structured "
        "answer to the original question. Reference the sub-questions where "
        "appropriate, avoid redundancy, and include all relevant details."
    )

    user_template = "User query: {query}\n\nVerified Summaries:\n{combined_summaries}\n\nFinal Answer:"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
        ]
    )

    messages = prompt.format_messages(
        query=query, combined_summaries=combined_summaries
    )

    llm_parsed = llm.with_structured_output(schema=FinalAnswer)
    response = llm_parsed.invoke(messages)

    state["answer"] = response.final_answer.strip()

    return state
