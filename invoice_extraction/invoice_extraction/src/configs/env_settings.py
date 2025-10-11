import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass(frozen=True)
class EnvSettings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = (
        os.environ.get("MODEL_QWEN")
        or os.environ.get("MODEL_GPT_OSS")
        or os.getenv("MODEL_SMALL", "llama:latest")
    )
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    
env_settings = EnvSettings()

print(env_settings)