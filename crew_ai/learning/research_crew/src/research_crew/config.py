import os
from dotenv import load_dotenv

load_dotenv(".venv")


class Config:
    project_out: str = os.getenv("PROJECT_OUT", "data/outputs")
    vector_dir: str = os.getenv("VECTOR_DIR", "data/knowledge")

    ollama_base: str = os.getenv("OLLAMA_BASE_URL", "")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "")


config = Config()
