"""
JobForge AI — Market Report Schema.

Single source of truth for a week's public market intelligence content.
The LinkedIn carousel, the email digest, and marketforge.digital previously
derived their figures semi-manually from MarketAnalyzer method calls, which
is how the salary-median divergence (~£46k vs ~£73k) shipped inconsistently
across surfaces. Everything downstream should now consume one validated
MarketReport instead of re-deriving figures per surface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MetricDelta(BaseModel):
    metric: str
    weeks: int
    current: float | None = None
    previous: float | None = None
    abs_change: float | None = None
    pct_change: float | None = None
    direction: Literal["up", "down", "flat", "unknown"] = "unknown"


class SkillTrajectory(BaseModel):
    weekly_counts: list[int] = Field(default_factory=list)
    trend: Literal["Accelerating", "Cooling", "Stable", "New"] = "Stable"
    slope: float = 0.0
    r_squared: float = 0.0


class SalaryPercentiles(BaseModel):
    n: int = 0
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None


class SalaryDivergence(BaseModel):
    diverges: bool = False
    weekly_median: float | None = None
    rolling_median: float | None = None
    pct_diff: float | None = None


class ReportMetadata(BaseModel):
    generated_at: datetime
    window_days: int
    total_jobs: int
    divergence_flagged: bool = False


class MarketReport(BaseModel):
    """
    Schema-validated snapshot of a week's market intelligence.

    Field groups mirror MarketAnalyzer's public methods 1:1 — see
    analytics/market_analyzer.py::build_market_report() for the assembly.
    """

    metadata: ReportMetadata

    top_skills: list[tuple[str, int]] = Field(default_factory=list)
    work_model: dict[str, int] = Field(default_factory=dict)
    sponsorship: dict[str, object] = Field(default_factory=dict)
    sponsor_register: dict[str, object] = Field(default_factory=dict)
    startup_ratio: dict[str, object] = Field(default_factory=dict)
    top_companies: list[tuple[str, int]] = Field(default_factory=list)
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    score_trend: list[dict] = Field(default_factory=list)
    cv_variants: dict[str, int] = Field(default_factory=dict)

    salary: dict[str, object] = Field(default_factory=dict)
    salary_percentiles: SalaryPercentiles = Field(default_factory=SalaryPercentiles)
    salary_by_category: dict[str, SalaryPercentiles] = Field(default_factory=dict)
    salary_by_seniority: dict[str, SalaryPercentiles] = Field(default_factory=dict)
    salary_divergence: SalaryDivergence = Field(default_factory=SalaryDivergence)

    deltas: dict[str, MetricDelta] = Field(default_factory=dict)
    skill_trajectories: dict[str, SkillTrajectory] = Field(default_factory=dict)
    rising_cooling_skills: dict[str, list[str]] = Field(default_factory=dict)
    role_category_distribution: dict[str, int] = Field(default_factory=dict)
    geographic_distribution: list[tuple[str, int]] = Field(default_factory=list)
    company_stage_distribution: dict[str, int] = Field(default_factory=dict)

    skill_co_occurrence: list[dict] = Field(default_factory=list)
    posting_persistence: dict[str, dict] = Field(default_factory=dict)
    salary_by_skill: dict[str, dict] = Field(default_factory=dict)
    work_model_trend: dict[str, list[int]] = Field(default_factory=dict)

    funnel: dict = Field(default_factory=dict)
