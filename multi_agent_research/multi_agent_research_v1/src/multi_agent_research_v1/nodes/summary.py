from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from multi_agent_research_v1.models.schemas import Summary
from multi_agent_research_v1.core.state import ResearchState


def summary_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Combine the contents of all documents into a single context string."""
    system_template = (
        "You are a summarisation assistant. Given a research sub-question "
        "and a context consisting of relevant passages, write a factual, "
        "concise summary that directly answers the question. Use only the "
        "information contained in the context; do not hallucinate."
    )

    llm_parsed = llm.with_structured_output(schema=Summary)

    summaries = {}
    for question, docs in state.get("retrieved_docs", {}).items():
        context = "\n\n".join([doc.page_content for doc in docs])

        user_template = "Question: {question}\nContext:\n{context}\nSummary:"

        prompt = ChatPromptTemplate(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template),
            ]
        )
        messages = prompt.format_messages(question=question, context=context)

        response = llm_parsed.invoke(messages)

        summaries[question] = response.summary

    state["summaries"] = summaries

    return state
