from duckdb import query
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import ToolNode

def create_tools(db, llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    list_tool = next(
        t for t in tools if t.name == "sql_db_list_tables"
    )

    schema_tool = next(
        t for t in tools if t.name == "sql_db_schema"
    )

    query_tool = next(
        t for t in tools if t.name == "sql_db_query"
    )
    
    query_check_tool = next(
        t for t in tools if t.name == "sql_db_query_checker"
    )

    list_node = ToolNode([list_tool])
    schema_node = ToolNode([schema_tool])
    query_node = ToolNode([query_tool])

    return (list_node, schema_node, query_node), query_check_tool