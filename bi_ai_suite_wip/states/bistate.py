from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langchain.schema import BaseMessage
import polars as pl


@dataclass
class BiState:
    messages: List[BaseMessage]
    data_sources: Dict[str, Any]
    raw_data: Dict[str, pl.DataFrame]
    analysis_results: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    reports: List[Dict[str, Any]]
    current_agent: Optional[str]
    task_completed: bool = False
