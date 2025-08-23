from dotenv import load_dotenv
import os
from dataclasses import dataclass

load_dotenv(".env")


@dataclass(frozen=True)
class Settings:
    model_url: str = os.getenv("OLLAMA_URL")
    llm_model: str = os.getenv("MODEL")
    embed_model: str = os.getenv("EMBED_MODEL")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY")

    max_iters: int = int(os.getenv("MAX_ITERS", 8))
    temperature: float = float(os.getenv("TEMPERATURE", 0.2))
    thread_ns: str = os.getenv("THREAD_NS", "demo")
    
    
SETTINGS = Settings()