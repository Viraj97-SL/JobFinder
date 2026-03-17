"""
JobForge AI — Job Data Models.

Every job flows through three stages:
  RawJob (Scout) → ScoredJob (Matchmaker) → TailoredJob (Tailor/Dispatcher)
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class RawJob(BaseModel):
    """A job as discovered by the Scout Agent — source-agnostic normalised schema."""

    job_id: str = Field(description="Source-specific unique identifier")
    title: str
    company: str
    location: str
    salary_min: float | None = None
    salary_max: float | None = None
    description: str = Field(max_length=8000, description="Full JD text")
    url: str
    source: str = Field(description="Connector name: adzuna, reed, wellfound, etc.")
    posted_date: date | None = None
    company_stage: Literal[
        "seed", "series_a", "series_b", "series_c", "series_d", "series_e",
        "series_f", "series_g", "growth", "enterprise", "nonprofit",
        "recently_funded", "acquired", "public", "government", "unknown"
    ] | None = None
    work_model: Literal["remote", "hybrid", "onsite", "unknown"] | None = None
    is_startup: bool = False

    # ── Visa Intelligence ──
    offers_sponsorship: bool | None = None       # True if JD explicitly offers visa sponsorship
    citizens_only: bool | None = None            # True if JD says "UK citizens only"
    sponsorship_signals: list[str] = Field(
        default_factory=list,
        description="Extracted phrases: 'visa sponsorship available', 'Skilled Worker', etc."
    )

    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def dedup_hash(self) -> str:
        """Stable hash for cross-source, cross-run deduplication."""
        raw = f"{self.title.lower().strip()}|{self.company.lower().strip()}|{self.location.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @computed_field
    @property
    def salary_display(self) -> str:
        if self.salary_min and self.salary_max:
            return f"£{self.salary_min:,.0f}–£{self.salary_max:,.0f}"
        elif self.salary_min:
            return f"From £{self.salary_min:,.0f}"
        elif self.salary_max:
            return f"Up to £{self.salary_max:,.0f}"
        return "Not Disclosed"


class ScoredJob(BaseModel):
    """A job after Matchmaker evaluation — includes match score and reasoning."""

    job: RawJob

    # ── Composite Score ──
    overall_score: float = Field(ge=0, le=100, description="Weighted composite 0–100")

    # ── Dimension Scores ──
    technical_skills_score: float = Field(ge=0, le=100)
    domain_experience_score: float = Field(ge=0, le=100)
    seniority_fit_score: float = Field(ge=0, le=100)
    location_score: float = Field(ge=0, le=100)
    visa_score: float = Field(ge=0, le=100)
    role_alignment_score: float = Field(ge=0, le=100)

    # ── LLM Reasoning ──
    reasoning: str = Field(description="2–3 sentence match justification")
    key_matching_skills: list[str] = Field(max_length=7)
    key_gaps: list[str] = Field(default_factory=list, description="Skills JD wants but CV lacks")
    transferable_highlights: list[str] = Field(
        default_factory=list,
        description="Cross-domain strengths to emphasise (e.g. MAS Holdings for supply chain)"
    )

    # ── Tailor Routing ──
    recommended_cv_variant: Literal["ai_engineer", "data_scientist", "ml_engineer"] = "ai_engineer"

    @computed_field
    @property
    def visa_tag(self) -> str:
        """Human-readable visa status for the Excel digest."""
        if self.job.offers_sponsorship:
            return "✓ Sponsorship"
        elif self.job.citizens_only:
            return "⚠ UK Citizens Only"
        return "—"

    @computed_field
    @property
    def tier(self) -> Literal["gold", "silver", "bronze"]:
        if self.overall_score >= 85:
            return "gold"
        elif self.overall_score >= 75:
            return "silver"
        return "bronze"


class TailoredJob(BaseModel):
    """Final output: scored job + tailored CV metadata."""

    scored_job: ScoredJob
    cv_variant_used: str
    cv_pdf_path: str
    cv_pdf_filename: str
    tailoring_notes: str = ""
    hallucination_check_passed: bool = True
    retry_count: int = 0
