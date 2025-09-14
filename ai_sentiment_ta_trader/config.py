import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")


class Config(BaseModel):
    ollama_base_url: str = os.getenv("BASE_URL")
    ollama_model: str = os.getenv("MODEL_LARGE") or os.getenv("MODEL_SMALL")


class Settings(BaseModel):
    max_news: int = 8
    yf_period: str = "1y"
    yf_interval: str = "1d"


config = Config()
settings = Settings()
