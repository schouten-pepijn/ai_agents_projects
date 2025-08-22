from dotenv import load_dotenv
from typing import Sequence, Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os

load_dotenv(".env")

MODEL = os.environ["MODEL"]
# MODEL = os.environ["MODEL_EXT"]
BASE_URL = os.environ["BASE_URL"]

model = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.1
)

prompt_template = ChatPromptTemplate(
    [
       ("system",
        "You are a helpful assistant. "\
            "You speak with an {accent} accent. " \
             "Answer all questions to the best of your ability. " \
       ),
       MessagesPlaceholder(
           variable_name="messages"
       )
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    accent: str
    

workflow = StateGraph(state_schema=State)


def call_model(state: State):
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=1024,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    prompt = prompt_template.invoke(
        {
            "messages": trimmed_messages,
            "accent": state["accent"]
        }
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
workflow.add_edge("model", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {
    "configurable": {"thread_id": "aabc123"}
}

query = "Hi I'm Pepijn."
accent = "italian"

input_messages = [HumanMessage(content=query)]
output = app.invoke(
    {
        "messages": input_messages,
        "accent": accent
    },
    config
)
output["messages"][-1].pretty_print()

input_messages = [HumanMessage(content="What is my name?")]
output = app.invoke(
    {
        "messages": input_messages,
        "accent": accent
    },
    config
)
output["messages"][-1].pretty_print()