import logging
from langchain_core.documents import Document
from multi_agent_research_v2.config.config import WorkflowConfig
from multi_agent_research_v2.utils.vectorstore import VectorStoreManager
from multi_agent_research_v2.utils.llm import initialize_llm
from multi_agent_research_v2.core.workflow import build_workflow
from multi_agent_research_v2.utils.logging_config import setup_logging
from multi_agent_research_v2.core.state import initialize_research_state

logger = setup_logging(logging.DEBUG)


def demo() -> None:

    docs = [
        Document(
            page_content=(
                "LangGraph is a framework for building stateful, agentic applications "
                "with language models. It extends LangChain with support for cyclic "
                "graphs, state persistence, and complex control flows, enabling developers "
                "to create sophisticated multi-agent systems."
            ),
            metadata={"source": "langgraph_docs", "section": "introduction"},
        ),
        Document(
            page_content=(
                "Multi-agent research workflows decompose complex queries into specialized "
                "tasks handled by different agents. Each agent focuses on a specific "
                "capability such as planning, retrieval, summarization, or verification. "
                "This modular architecture improves scalability and maintainability."
            ),
            metadata={"source": "research_patterns", "section": "architecture"},
        ),
        Document(
            page_content=(
                "Conditional routing in LangGraph allows workflows to make dynamic decisions "
                "based on intermediate results. Routes can loop back for refinement, skip "
                "unnecessary steps, or branch to alternative paths. This enables adaptive, "
                "intelligent workflows that respond to quality assessments."
            ),
            metadata={"source": "langgraph_docs", "section": "routing"},
        ),
        Document(
            page_content=(
                "Quality gates are checkpoints that assess intermediate outputs against "
                "defined thresholds. If quality is insufficient, the workflow can trigger "
                "refinement loops. This iterative approach ensures high-quality final "
                "outputs in production systems."
            ),
            metadata={"source": "best_practices", "section": "quality"},
        ),
        Document(
            page_content=(
                "Vector stores like FAISS enable efficient similarity search over document "
                "embeddings. In RAG systems, they retrieve relevant context for language "
                "model prompts, grounding responses in factual information and reducing "
                "hallucination."
            ),
            metadata={"source": "rag_fundamentals", "section": "retrieval"},
        ),
    ]

    # Initialize configuration
    config = WorkflowConfig()

    # Initialize vector store manager and create vector store
    vector_store_manager = VectorStoreManager(config)
    vector_store = vector_store_manager.prepare_vector_store(docs)

    logger.info(f"Vector store created successfully with {len(docs)} documents")

    llm = initialize_llm(
        model_name=config.model_name,
        base_url=config.base_url,
        temperature=config.temperature,
    )

    workflow = build_workflow(
        llm=llm,
        vector_store=vector_store,
        config=config,
    )
    app = workflow.compile()

    query = "Explain LangGraph's approach to conditional routing and quality gates."
    initial_state = initialize_research_state(query)

    logger.info(f"Starting research workflow for query: {query}")

    result_state = app.invoke(initial_state)

    logger.info("Research workflow completed successfully")

    print("\nSUB-QUESTIONS:")
    for i, sq in enumerate(result_state["sub_questions"], 1):
        print(f"  {i}. {sq}")

    print("\nQUALITY SCORES:")
    for node, score in result_state["quality_scores"].items():
        print(f"  {node}: {score:.2f}")

    print("\nROUTING HISTORY:")
    for event in result_state["routing_history"]:
        print(f"  → {event}")

    print("\nSUMMARIES:")
    for q, s in result_state["summaries"].items():
        print(f"\n  Q: {q}")
        print(f"  A: {s}\n")

    if result_state and result_state.get("answer"):
        print("\n" + "=" * 50)
        print("FINAL RESEARCH ANSWER:")
        print("=" * 50)
        print(result_state["answer"])
        print("=" * 50)

    else:
        logger.warning("No final answer found in result state")

        if result_state:
            print(f"\nWorkflow status: {result_state.get('status')}")
            print(f"Errors: {result_state.get('errors', [])}")


if __name__ == "__main__":
    demo()
