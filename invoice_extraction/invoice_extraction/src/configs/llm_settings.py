from configs.env_settings import env_settings
from langchain_ollama import ChatOllama, OllamaEmbeddings


def get_ollama_chat_client():
    return ChatOllama(
        base_url=env_settings.ollama_base_url,
        model=env_settings.ollama_model,
        temperature=0.1,
    )
    
def get_ollama_embed_client():
    return OllamaEmbeddings(
        base_url=env_settings.ollama_base_url,
        model=env_settings.embed_model,
        temperature=0.0,
    )