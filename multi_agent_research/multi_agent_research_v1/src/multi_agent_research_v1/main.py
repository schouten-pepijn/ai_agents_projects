import os
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Optional
from pydantic import BaseModel, Field
from langchain_core.documents.base import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores.faiss import FAISS
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(".env")


class ResearchState(TypedDict):
    """State for intermediate and final values in the flow."""

    query: str
    sub_questions: List[str]
    retrieved_docs: Dict[str, List[Document]]
    summaries: Dict[str, str]
    answer: Optional[str]


def prepare_vector_store(documents: List[Document]) -> FAISS:
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL"), base_url=os.getenv("BASE_URL")
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def create_llm(temperature: float = 0.0) -> ChatOllama:

    return ChatOllama(
        model=os.getenv("MODEL_LARGE"),
        temperature=temperature,
        base_url=os.getenv("BASE_URL"),
    )


class SubQuestions(BaseModel):
    """Schema for the sub-questions output."""

    sub_questions: List[str] = Field(description="List of 2-4 focused sub-questions")


def planner_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Decompose the query intro a set of actionable sub-questions."""

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


def retrieval_node(
    state: ResearchState, vector_store: FAISS, k: int = 4
) -> ResearchState:
    """Retrieve relevant documents for each sub-question."""

    retrieved = {}
    for question in state.get("sub_questions", []):
        docs_and_scores = vector_store.similarity_search_with_score(question, k=k)
        docs = [doc for doc, score in docs_and_scores]
        retrieved[question] = docs

    state["retrieved_docs"] = retrieved
    return state


class Summary(BaseModel):
    """Schema for the summary output."""

    summary: str = Field(description="A concise summary of the research findings")


def summary_node(state: ResearchState, llm: ChatOllama) -> ResearchState:
    """Combine the contents of all documents into a single context string."""

    system_template = (
        "You are a summarisation assistant. Given a research sub‑question "
        "and a context consisting of relevant passages, write a factual, "
        "concise summary that directly answers the question. Use only the "
        "information contained in the context; do not hallucinate."
    )

    llm_parsed = llm.with_structured_output(schema=Summary)

    summaries = {}
    for question, docs in state.get("retrieved_docs", {}).items():
        context = "\n\n".join([doc.page_content for doc in docs])

        user_template = "Question: {question}\nContext:\n{context}\nSummary:"

        prompt = ChatPromptTemplate.from_messages(
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


class Assessment(BaseModel):
    """Schema for the assessment output."""

    assessment: str = Field(
        description="Assessment of the summary's completeness and accuracy"
    )


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

        assessment = response.summary.strip()
        if assessment.upper() == "OK":
            verified[question] = summary

        else:
            verified[question] = summary + "\n\n[Reviewer comments: " + assessment + "]"

    state["summaries"] = verified

    return state


class FinalAnswer(BaseModel):
    """Schema for the final answer output."""

    final_answer: str = Field(
        description="A comprehensive final answer to the original user question"
    )


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


def build_workflow(llm: ChatOllama, vector_store: FAISS) -> StateGraph:
    """Build the research workflow as a sequence of nodes."""

    workflow: StateGraph = StateGraph(state_schema=ResearchState)

    def planner_wrapper(state: ResearchState) -> ResearchState:
        return planner_node(state, llm)

    def retriever_wrapper(state: ResearchState) -> ResearchState:
        return retrieval_node(state, vector_store)

    def summarizer_wrapper(state: ResearchState) -> ResearchState:
        return summary_node(state, llm)

    def verifier_wrapper(state: ResearchState) -> ResearchState:
        return verification_node(state, llm)

    def synthesiser_wrapper(state: ResearchState) -> ResearchState:
        return synthesis_node(state, llm)

    workflow.add_node("planner", planner_wrapper)
    workflow.add_node("retriever", retriever_wrapper)
    workflow.add_node("summarizer", summarizer_wrapper)
    workflow.add_node("verifier", verifier_wrapper)
    workflow.add_node("synthesizer", synthesiser_wrapper)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "summarizer")
    workflow.add_edge("summarizer", "verifier")
    workflow.add_edge("verifier", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


def demo() -> None:
    """Demonstrate the multi-agent research workflow."""

    docs = [
        Document(
            page_content=(
                "LangGraph is a framework for building agentic applications. It "
                "extends LangChain with support for long-running, stateful graphs and "
                "cyclical control flows, enabling developers to create custom multi-agent "
                "systems."
            ),
            metadata={"source": "knowledge_base"},
        ),
        Document(
            page_content=(
                "In multi-agent research workflows, each agent specialises in a "
                "distinct task such as planning, retrieval or synthesis. This separation "
                "of concerns improves modularity and scalability."
            ),
            metadata={"source": "knowledge_base"},
        ),
        Document(
            page_content=(
                "Vector stores like FAISS index document embeddings to enable efficient "
                "similarity search. Retrieval-augmented generation (RAG) systems "
                "combine vector search with language models to provide grounded answers."
            ),
            metadata={"source": "knowledge_base"},
        ),
        Document(
            page_content=(
                "Planning sub-tasks before retrieval helps break down complex queries "
                "into manageable pieces, reducing hallucination and improving answer quality."
            ),
            metadata={"source": "knowledge_base"},
        ),
    ]

    # Prepare vector store
    vector_store = prepare_vector_store(docs)

    # Instantiate language model
    llm = create_llm()

    # Build workflow
    workflow = build_workflow(llm, vector_store)

    # Initial research state
    initial_state: ResearchState = {
        "query": "How does LangGraph facilitate multi-agent research workflows?",
        "sub_questions": [],
        "retrieved_docs": {},
        "summaries": {},
        "answer": None,
    }

    # Execute the workflow
    result_state = workflow.invoke(initial_state)

    # Display results
    print("Sub-questions:\n", result_state["sub_questions"])

    print("\nSummaries:")

    for q, s in result_state["summaries"].items():
        print(f" - {q}: {s}\n")

    print("Final answer:\n", result_state["answer"])


if __name__ == "__main__":
    demo()
