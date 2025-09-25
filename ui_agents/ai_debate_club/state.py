from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class Role(Enum):
    PRO = "pro"
    CON = "con"
    MOD = "mod"

@dataclass
class Utterance:
    role: str
    content: str
    round: int

@dataclass
class DebateState:
    topic: str
    max_rounds: int
    round_idx: int = 0
    transcript: List[Utterance] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)
    winner_per_round: List[str] = field(default_factory=list)
    final_summary: Optional[str] = None
