from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    text: str
    structure: Dict[str, Any]
    images: List[bytes] = Field(default_factory=list)
    

class FieldSpec(BaseModel):
    name: str
    type_: str
    required: bool = True
    
    
class Candidate(BaseModel):
    value: Any
    confidence: float
    evidence: Optional[Dict[str, Any]] = None