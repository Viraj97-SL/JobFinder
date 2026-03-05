"""
JobForge AI — Scout Agent Query Planner.

Generates optimised search queries for each job source based on
target roles, skills, and startup focus areas.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchPlan:
    """The Scout Agent's execution plan — what to search, where, and why."""

    primary_queries: list[str] = field(default_factory=list)
    startup_queries: list[str] = field(default_factory=list)
    skill_specific_queries: list[str] = field(default_factory=list)
    nonprofit_queries: list[str] = field(default_factory=list)
    sources_to_query: list[str] = field(default_factory=list)

    @property
    def all_queries(self) -> list[str]:
        return self.primary_queries + self.startup_queries + self.skill_specific_queries + self.nonprofit_queries

    def for_source(self, source: str) -> list[str]:
        """Get queries optimised for a specific source."""
        if source in ("wellfound", "yc_startup"):
            return self.startup_queries + self.primary_queries[:3]
        elif source == "career_pages":
            return self.skill_specific_queries + self.primary_queries[:2]
        else:
            return self.primary_queries + self.startup_queries[:2]


def build_search_plan(
    target_roles: list[str] | None = None,
    location: str = "UK",
    extra_skills: list[str] | None = None,
) -> SearchPlan:
    """
    Build a comprehensive search plan based on user preferences.

    This is deterministic — no LLM needed. The plan is based on
    the job_search_prefs.yaml configuration.
    """

    roles = target_roles or [
        "AI Engineer",
        "Machine Learning Engineer",
        "ML Engineer",
        "LLM Engineer",
        "Data Scientist",
        "Computer Vision Engineer",
    ]

    locations = ["London", "UK", "UK remote"]

    # ── Primary Queries: role × location ──
    primary = []
    for role in roles:
        for loc in locations:
            primary.append(f"{role} {loc}")

    # ── Startup-Focused Queries ──
    startup = [
        "AI Engineer startup London",
        "ML Engineer seed stage UK",
        "Data Scientist Series A London",
        "AI Engineer early stage UK",
        "Machine Learning startup UK remote",
        "LLM Engineer founding team London",
        "AI startup hiring UK",
        "ML Engineer scale-up London",
    ]

    # ── Skill-Specific Queries (for career pages & niche sources) ──
    skills = extra_skills or [
        "LangGraph", "LangChain", "Computer Vision",
        "NLP", "PyTorch", "multi-agent systems",
    ]
    skill_queries = [f"{skill} engineer {location}" for skill in skills]

    # ── Nonprofit / Mission-Driven ──
    nonprofit = [
        "Data Scientist nonprofit UK",
        "AI for Good engineer London",
        "Machine Learning charity UK",
        "Data Scientist social impact London",
        "AI Engineer healthcare UK",
    ]

    return SearchPlan(
        primary_queries=primary,
        startup_queries=startup,
        skill_specific_queries=skill_queries,
        nonprofit_queries=nonprofit,
        sources_to_query=[
            "adzuna", "reed", "wellfound", "linkedin_proxy",
            "indeed_proxy", "career_pages",
        ],
    )
