# SQL Agent with Ollama

A LangGraph-powered SQL agent that can interact with a SQLite database using natural language queries via Ollama.

## Features

- **Natural Language to SQL**: Convert natural language questions to SQL queries
- **Read-only Safety**: Only SELECT queries are allowed to prevent data modification
- **Schema Discovery**: Automatically explores database structure when needed
- **Tool-based Architecture**: Uses dedicated tools for listing tables, describing schemas, and executing queries
- **Memory Persistence**: Maintains conversation context using LangGraph checkpointing

## Prerequisites

1. **Python 3.10+** with the required packages installed
2. **Ollama** running locally at `http://localhost:11434`
3. A compatible model (e.g., `llama3.2`) available in Ollama

## Installation

1. Install the required dependencies:
```bash
pip install langchain-community langchain-core langchain-ollama langgraph sqlalchemy
```

2. Update the model name in `config.py` if needed:
```python
MODEL = "llama3.2"  # or your preferred model
```

## Usage

### Basic Usage

```python
from sql_agent import run_agent

# Ask questions about the database
answer = run_agent("Which customers from NL placed orders?")
print(answer)

answer = run_agent("Show me the top 3 orders by amount with customer details")
print(answer)

answer = run_agent("What tables are available?")
print(answer)
```

### Running the Agent

```bash
# Test the database setup
python test_db.py

# Initialize the agent (without auto-running examples)
python sql_agent.py
```

## Database Schema

The agent comes with a demo SQLite database containing:

### Customers Table
- `customer_id` (INTEGER PRIMARY KEY)
- `name` (TEXT NOT NULL)
- `country` (TEXT NOT NULL)

### Orders Table
- `order_id` (INTEGER PRIMARY KEY)
- `customer_id` (INTEGER NOT NULL, FOREIGN KEY)
- `order_date` (TEXT NOT NULL)
- `amount` (REAL NOT NULL)
- `status` (TEXT NOT NULL)

Sample data includes customers from different countries (NL, DE, BE) and their corresponding orders.

## Example Questions

- "Which customers from NL placed orders, and what is their total amount?"
- "Give me the top 3 orders by amount with customer name and status."
- "Show me the tables and the columns of each."
- "What is the average order amount by country?"
- "Which customers have pending orders?"

## Architecture

### Components

1. **config.py**: Database and model configuration
2. **init_db.py**: Database initialization and sample data
3. **prompts.py**: System prompt for the SQL agent
4. **sql_agent.py**: Main agent implementation with tools and LangGraph setup
5. **test_db.py**: Database functionality testing

### Tools

- **list_tables**: Get all available table names
- **describe_tables**: Get detailed schema information for specific tables
- **query_sql**: Execute read-only SELECT queries

### Safety Features

- SQL injection prevention through parameterized queries
- Read-only enforcement (only SELECT statements allowed)
- Query validation to block modification operations

## Configuration

Edit `config.py` to customize:

```python
DB_PATH = "database/demo.sqlite"        # Database file path
MODEL_URL = "http://localhost:11434"    # Ollama server URL
MODEL = "llama3.2"                      # Model name
```

## Troubleshooting

1. **"ModuleNotFoundError"**: Install the required dependencies
2. **Ollama connection errors**: Ensure Ollama is running and the model is available
3. **Database errors**: Run `test_db.py` to verify database setup
4. **Model timeout**: Check if the specified model is downloaded in Ollama

## Development

To modify the agent:

1. Update the system prompt in `prompts.py`
2. Add new tools in `sql_agent.py`
3. Modify the database schema in `init_db.py`
4. Adjust the agent configuration in the `create_react_agent` call

## License

This project is for educational and demonstration purposes.
