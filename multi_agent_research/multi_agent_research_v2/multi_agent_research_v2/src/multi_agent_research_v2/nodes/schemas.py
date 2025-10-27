"""Pydantic schemas for structured LLM outputs."""

from typing import List
from pydantic import BaseModel, Field


class SubQuestionsOutput(BaseModel):
    """Schema for planner node output."""

    sub_questions: List[str] = Field(
        description="List of 2-5 focused sub-questions that decompose the main research query",
        min_length=2,
        max_length=5,
    )


class QualityAssessmentOutput(BaseModel):
    """Schema for quality assessment output."""

    score: float = Field(description="Quality score from 0 to 10", ge=0, le=10)
    reasoning: str = Field(description="Brief explanation of the quality assessment")


class SummaryQualityOutput(BaseModel):
    """Schema for summary quality assessment."""

    score: float = Field(description="Quality score from 0 to 10", ge=0, le=10)
    issues: List[str] = Field(
        description="List of identified issues with the summary",
        default_factory=list,
    )


class VerificationOutput(BaseModel):
    """Schema for verification node output."""

    status: str = Field(description="Verification status: 'pass' or 'fail'")
    score: float = Field(description="Verification score from 0 to 10", ge=0, le=10)
    feedback: str = Field(description="Detailed feedback on the verification result")


class SummaryOutput(BaseModel):
    """Schema for summarizer node output."""

    summary: str = Field(
        description="A focused 2-4 sentence summary that answers the question based on the context"
    )


class SynthesisOutput(BaseModel):
    """Schema for synthesizer node output."""

    answer: str = Field(
        description="A comprehensive, well-structured final answer that integrates all research findings"
    )
