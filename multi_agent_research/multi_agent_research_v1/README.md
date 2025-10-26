# Multi-Agent Research Workflow

A multi-agent research system using LangGraph and LangChain that breaks down complex research questions into manageable sub-questions, retrieves relevant information, and synthesizes findings into comprehensive answers.

## Features

- **Query Decomposition**: Breaks research questions into focused sub-questions
- **Intelligent Retrieval**: Retrieves relevant documents for each sub-question using FAISS vector store
- **Quality Verification**: Validates summaries for completeness and accuracy
- **Iterative Refinement**: Automatically reformulates failed queries and re-retrieves (max 2 iterations)
- **Smart Query Expansion**: Replaces only failed questions, preserving successful ones
- **Final Synthesis**: Combines verified summaries into a coherent final answer

## Workflow

```text
Query → Planner → Retriever → Summarizer → Verifier
                                              ↓
                                        (Issues Found?)
                                              ↓
                                        Query Expander → Retriever (loop)
                                              ↓
                                         (Max Iterations?)
                                              ↓
                                         Synthesizer → Answer
```

## Quick Start

### Installation

```bash
poetry install
```

### Running the Demo

```bash
poetry run python src/multi_agent_research_v1/main.py
```

## Configuration

Edit `src/multi_agent_research_v1/main.py` to:

- Change the research query
- Adjust `max_iterations` (default: 2)
- Modify the knowledge base documents

## Logging

The system provides detailed logging of the workflow execution:

- `INFO` level: Node transitions, iteration counts, verification results
- Configure in `main.py` by changing `setup_logging()` level

## Architecture

```text
src/multi_agent_research_v1/
├── main.py              # Entry point
├── core/
│   ├── workflow.py      # LangGraph workflow definition
│   └── state.py         # Research state schema
├── nodes/               # Individual agent nodes
│   ├── planner.py       # Query decomposition
│   ├── retrieval.py     # Document retrieval
│   ├── summary.py       # Summary generation
│   ├── verification.py  # Quality verification
│   ├── query_expansion.py  # Query reformulation
│   └── synthesis.py     # Final answer synthesis
├── models/
│   └── schemas.py       # Data schemas
└── utils/
    ├── llm.py           # LLM initialization
    ├── vector_store.py  # FAISS setup
    └── logging_config.py # Logging configuration
```

## Key Components

- **LangGraph**: Orchestrates the multi-agent workflow with state management
- **LangChain**: Provides LLM integration and prompting utilities
- **FAISS**: Vector search for document retrieval
- **Ollama**: Local LLM support (configurable)

## License

MIT
