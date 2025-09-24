"""Configuration module for PDF RAG chatbot."""

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(".env")


@dataclass(frozen=True)
class Config:
    MODEL: str = (
        os.getenv("MODEL_LARGE")
        or os.getenv("MODEL_MEDIUM")
        or os.getenv("MODEL_SMALL")
        or "llama2"
    )
    BASE_URL: str = os.environ["BASE_URL"]
    EMBED_MODEL: str = os.environ["EMBED_MODEL"]

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    MAX_RETRIEVAL_DOCS: int = 10
    MAX_HIGHLIGHT_BLOCKS: int = 5
    SIMILARITY_WINDOW_SENTENCES: int = 1

    MAX_QUESTION_LENGTH: int = 500

    LLM_TEMPERATURE: float = 0.1
