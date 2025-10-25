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


class FinalAnswer(BaseModel):
    """Schema for the final answer output."""

    final_answer: str = Field(
        description="A comprehensive final answer to the original user question"
    )
