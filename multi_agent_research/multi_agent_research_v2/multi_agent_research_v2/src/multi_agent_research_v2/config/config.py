import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv(".env")


@dataclass
class WorkflowConfig:
    """Centralized configuration for the research workflow."""

    model_name: str = os.getenv("MODEL_LARGE", None) or os.getenv(
        "MODEL_SMALL", "llama3.1:8b"
    )
    base_url: str = os.environ["BASE_URL"]
    embed_model: str = os.environ["EMBED_MODEL"]
    temperature: float = 0.0
    max_retries: int = 3

    retrieval_k: int = 4
    similarity_threshold: float = 0.7

    min_quality_score: float = 0.7
    max_refinement_iterations: int = 2

    min_subquestions: int = 2
    max_subquestions: int = 5

    min_answer_length: int = 100
    max_answer_length: int = 2000

    persist_directory: str = "./vector_store"


class NodeDecision(str, Enum):
    """Enum for routing decisions."""

    CONTINUE = "continue"
    REFINE = "refine"
    SKIP = "skip"
    FAIL = "fail"
    COMPLETE = "complete"


class QualityLevel(str, Enum):
    """Enum for quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"
