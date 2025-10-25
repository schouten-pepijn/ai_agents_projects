# Multi-Agent Research Workflow

This project implements a multi-agent research workflow using LangChain and related technologies. The workflow is designed to decompose research questions into actionable sub-questions, retrieve relevant documents, summarize findings, verify the accuracy of summaries, and synthesize a final answer.

## Project Structure

```
multi_agent_research_v1
├── src
│   └── multi_agent_research_v1
│       ├── __init__.py
│       ├── main.py               # Entry point for the application
│       ├── core
│       │   ├── __init__.py
│       │   ├── state.py          # Manages the research state
│       │   └── workflow.py       # Builds and manages the workflow
│       ├── nodes
│       │   ├── __init__.py
│       │   ├── planner.py        # Decomposes main question into sub-questions
│       │   ├── retrieval.py      # Retrieves documents for sub-questions
│       │   ├── summary.py        # Summarizes retrieved documents
│       │   ├── verification.py    # Assesses summary completeness and accuracy
│       │   └── synthesis.py      # Combines summaries into a final answer
│       ├── models
│       │   ├── __init__.py
│       │   └── schemas.py        # Defines data models and schemas
│       └── utils
│           ├── __init__.py
│           ├── llm.py            # Utility functions for language model
│           └── vector_store.py   # Utility functions for vector store
├── tests
│   ├── __init__.py
│   ├── test_nodes.py            # Unit tests for nodes
│   └── test_workflow.py         # Unit tests for workflow logic
├── .env                          # Environment variables
├── .gitignore                    # Git ignore file
├── pyproject.toml                # Project configuration
└── README.md                     # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd multi_agent_research_v1
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables in the `.env` file. Ensure to include necessary API keys and model configurations.

## Usage

To run the application, execute the following command:
```
python -m multi_agent_research_v1.main
```

This will initiate the multi-agent research workflow, processing the predefined query and displaying the results.

## Testing

To run the tests, use:
```
pytest tests/
```

This will execute all unit tests defined in the `tests` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.