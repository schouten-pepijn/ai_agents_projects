from dotenv import load_dotenv
from langchain_core.documents.base import Document
from multi_agent_research_v1.core.state import ResearchState
from multi_agent_research_v1.core.workflow import build_workflow
from multi_agent_research_v1.utils.llm import create_llm
from multi_agent_research_v1.utils.vector_store import prepare_vector_store

load_dotenv(".env")


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
                "In multi-agent research workflows, each agent specializes in a "
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
