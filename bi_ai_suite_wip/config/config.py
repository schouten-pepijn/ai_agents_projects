import os
from typing import Optional, Dict
from dataclasses import dataclass
from dotenv import load_dotenv
from enum import Enum

load_dotenv()


@dataclass
class LlmConfig:
    llm_base_url: str = os.getenv("BASE_URL")
    llm_model: str = os.getenv("MODEL_LARGE", "MODEL_SMALL")


class DataFormatConfig(Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"

@dataclass
class DataSourceConfig:
    name: str
    source_type: str
    endpoint: str
    credentials: Optional[Dict] = None
    parameters: Optional[Dict] = None
    