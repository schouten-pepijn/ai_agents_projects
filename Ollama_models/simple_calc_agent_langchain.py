import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")

llm = ChatOllama(
    base_url=OLLAMA_URL,
    model=MODEL,
    temperature=0.1,
    num_ctx=4096,
    seed=87,
)


class CalcInput(BaseModel):
    expression: str = Field(..., description="Arithmetic using digits and + - * / ( ) .")

def _calc(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    
    if not set(expression) <= allowed:
        return "Invalid characters"
    
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    
    except Exception as e:
        return f"Error: {e}"


calculator = StructuredTool.from_function(
    name="calculator",
    description="Evaluate arithmetic expressions exactly. Use for any math.",
    func=_calc,              
    args_schema=CalcInput,
)

tools = [calculator]


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise AI assistant. Use tools when helpful and answer concisely."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


out = executor.invoke({
    "input": "Compute 23*47 and then describe the bias-variance trade-off in one sentence."
})
print("\n--- ANSWER ---\n", out["output"])
