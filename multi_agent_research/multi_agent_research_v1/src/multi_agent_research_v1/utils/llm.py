from langchain_ollama.chat_models import ChatOllama
import os


def create_llm(temperature: float = 0.0) -> ChatOllama:
    """Create and return a ChatOllama instance configured with environment variables."""

    return ChatOllama(
        model=os.getenv("MODEL_LARGE"),
        temperature=temperature,
        base_url=os.getenv("BASE_URL"),
    )
