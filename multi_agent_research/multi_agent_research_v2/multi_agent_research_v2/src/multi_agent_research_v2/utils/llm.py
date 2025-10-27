import logging
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

logger = logging.getLogger("multi_agent_research")


def initialize_llm(
    model_name: str, base_url: str, temperature: float = 0.0
) -> ChatOllama:
    """Initialize and return a ChatOllama LLM instance."""
    logger.info(f"Initializing LLM with model '{model_name}' at '{base_url}'")
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
    return llm


def initialize_embeddings(model_name: str, base_url: str) -> OllamaEmbeddings:
    """Initialize and return OllamaEmbeddings instance."""
    logger.info(f"Initializing embeddings with model '{model_name}' at '{base_url}'")
    embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
    return embeddings
