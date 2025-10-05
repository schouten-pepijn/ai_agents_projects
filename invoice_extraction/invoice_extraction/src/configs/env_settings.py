import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass(frozen=True)
class EnvSettings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL")
    ollama_model: str = os.getenv("MODEL_LARGE", "MODEL_SMALL")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    
env_settings = EnvSettings()

print(env_settings)