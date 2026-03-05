"""
JobForge AI — Scoring & Match Models.

Structured output schemas that the Matchmaker LLM must produce.
Using Pydantic V2 for strict validation of LLM JSON responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchScore(BaseModel):
    """
    Structured JSON output from the Matchmaker LLM.

    The LLM receives: (job_description, skill_inventory, scoring_rubric)
    and must return this exact schema. Pydantic validates on parse.
    """

    overall_score: float = Field(ge=0, le=100)
    technical_skills_score: float = Field(ge=0, le=100)
    domain_experience_score: float = Field(ge=0, le=100)
    seniority_fit_score: float = Field(ge=0, le=100)
    location_score: float = Field(ge=0, le=100)
    visa_score: float = Field(ge=0, le=100)
    role_alignment_score: float = Field(ge=0, le=100)

    reasoning: str = Field(
        min_length=20,
        max_length=500,
        description="2-3 sentence justification"
    )
    key_matching_skills: list[str] = Field(min_length=1, max_length=7)
    key_gaps: list[str] = Field(default_factory=list)
    transferable_highlights: list[str] = Field(default_factory=list)
    recommended_cv_variant: str = Field(
        pattern=r"^(ai_engineer|data_scientist|data_engineer)$"
    )


class MatchSummary(BaseModel):
    """Aggregate statistics for a single pipeline run."""

    total_scraped: int = 0
    total_after_dedup: int = 0
    total_prescreened: int = 0       # Passed embedding threshold
    total_scored: int = 0            # LLM-scored
    total_qualified: int = 0         # >= match_threshold
    average_score: float = 0.0
    highest_score: float = 0.0
    highest_score_company: str = ""
    score_distribution: dict[str, int] = Field(
        default_factory=lambda: {
            "90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "below_60": 0
        }
    )
    sponsoring_jobs_count: int = 0
    startup_jobs_count: int = 0
