import os
from dotenv import load_dotenv
from dataclasses import dataclass
from paths.paths import VECTORDB_DIR

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
    
    chroma_persist_directory: str = str(VECTORDB_DIR)
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "invoice_chunks")
    
    
env_settings = EnvSettings()

print(env_settings)