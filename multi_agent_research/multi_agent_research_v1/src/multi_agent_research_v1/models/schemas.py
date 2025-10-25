from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class SubQuestions(BaseModel):
    """Schema for the sub-questions output."""

    sub_questions: List[str] = Field(description="List of 2-4 focused sub-questions")


class Summary(BaseModel):
    """Schema for the summary output."""

    summary: str = Field(description="A concise summary of the research findings")


class Assessment(BaseModel):
    """Schema for the assessment output."""

    assessment: str = Field(
        description="Assessment of the summary's completeness and accuracy"
    )
    confidence: float = Field(
        description="Confidence score (0.0 to 1.0) regarding the assessment",
        ge=0.0,
        le=1.0,
    )
    needs_refinement: bool = Field(
        description="Indicates if the summary needs further refinement"
    )


class FinalAnswer(BaseModel):
    """Schema for the final answer output."""

    final_answer: str = Field(
        description="A comprehensive final answer to the original user question"
    )
