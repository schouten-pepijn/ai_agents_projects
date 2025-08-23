from langchain_ollama import OllamaEmbeddings, ChatOllama
from config import Config

def create_llm() -> ChatOllama:
    return ChatOllama(
        model=Config.MODEL,
        base_url=Config.BASE_URL,
        temperature=Config.LLM_TEMPERATURE,
    )

def create_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=Config.EMBED_MODEL,
        base_url=Config.BASE_URL,
    )

llm = create_llm()
embedder = create_embedder()
