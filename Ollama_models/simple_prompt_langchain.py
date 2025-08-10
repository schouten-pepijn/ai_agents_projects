import os

from dotenv import load_dotenv

from langchain_ollama import ChatOllama

load_dotenv(".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL")


llm = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_URL,
    temperature=0.1,
    num_ctx=4096,
    seed=87
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)