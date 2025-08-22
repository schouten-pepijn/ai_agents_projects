from langchain_ollama import ChatOllama
from config import MODEL, BASE_URL

def build_llm():
    llm = ChatOllama(
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.1,
    )
    return llm