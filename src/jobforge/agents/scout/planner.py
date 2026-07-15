"""
JobForge AI — Scout Agent Query Planner.

Generates optimised search queries for each job source based on
target roles, skills, and startup focus areas.

Sources and their query strategies:
  adzuna / reed / uk_gov_find_a_job → primary role × location queries
  wellfound                        → startup-focused + primary
  career_pages / hn_who_is_hiring   → skill-specific (used as keyword filters)
  ats_direct            → no queries needed (uses internal ATS token discovery)
  funding_news          → no queries needed (uses internal news queries)
  recruiter_boards      → no queries needed (uses internal agency queries)
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
        return (
            self.primary_queries
            + self.startup_queries
            + self.skill_specific_queries
            + self.nonprofit_queries
        )

    def for_source(self, source: str) -> list[str]:
        """Get queries optimised for a specific source."""
        if source in ("wellfound", "yc_startup"):
            return self.startup_queries + self.primary_queries[:3]
        elif source in ("career_pages", "hn_who_is_hiring"):
            # Career pages and the HN thread both use queries as keyword
            # filters (not literal search terms) — skill-specific works best
            return self.skill_specific_queries + self.primary_queries[:3]
        elif source in ("ats_direct", "funding_news", "recruiter_boards"):
            # These connectors have their own internal query logic
            return []
        else:
            # adzuna, reed, indeed_proxy, linkedin_proxy, uk_gov_find_a_job
            return self.primary_queries + self.startup_queries[:3]


def build_search_plan(
    target_roles: list[str] | None = None,
    location: str = "UK",
    extra_skills: list[str] | None = None,
) -> SearchPlan:
    """
    Build a comprehensive search plan based on user preferences.

    Deterministic — no LLM needed. Covers:
    - Core AI/ML/DS role titles × UK location variants
    - Startup-stage specific queries
    - Deep skill-specific queries (for career pages / niche sources)
    - Mission-driven / nonprofit queries
    - Emerging / specialist role titles (2025 market)
    """

    roles = target_roles or [
        # Core titles
        "AI Engineer",
        "Machine Learning Engineer",
        "ML Engineer",
        "LLM Engineer",
        "Data Scientist",
        # Specialist titles growing in demand
        "Computer Vision Engineer",
        "NLP Engineer",
        "MLOps Engineer",
        "Research Scientist",
        "Applied Scientist",
        "AI Research Engineer",
        "Generative AI Engineer",
        "Foundation Model Engineer",
        "Multimodal AI Engineer",
    ]

    locations = ["London", "UK", "UK remote", "remote UK", "hybrid London"]

    # ── Primary Queries: role × location ──────────────────────────────────────
    primary: list[str] = []
    for role in roles[:8]:  # Top 8 roles × 5 locations = 40 queries
        for loc in locations:
            primary.append(f"{role} {loc}")

    # ── Startup-Focused Queries ────────────────────────────────────────────────
    startup = [
        # Stage signals
        "AI Engineer startup London seed round",
        "ML Engineer series A B London startup",
        "Data Scientist early stage UK startup",
        "AI Engineer founding team London",
        "Machine Learning startup UK remote hiring",
        "LLM Engineer founding engineer London",
        # YC / accelerator backed
        "YCombinator AI startup London engineer",
        "AI startup London accelerator backed hiring",
        # Specific company signals
        "AI scale-up London hiring engineers 2025 2026",
        "ML Engineer deep tech London startup",
        # Research-focused
        "AI research engineer London startup",
        "ML research scientist early stage UK",
        # Generative AI wave
        "Generative AI engineer London startup",
        "LLM product engineer startup UK",
        "AI agent engineer London startup hiring",
    ]

    # ── Skill-Specific Queries ─────────────────────────────────────────────────
    skills = extra_skills or [
        # Frameworks
        "LangGraph", "LangChain", "LlamaIndex", "PyTorch", "JAX",
        # Domains
        "Computer Vision", "NLP", "Reinforcement Learning", "MLOps",
        "Multi-agent systems", "RAG", "Fine-tuning", "RLHF",
        # Infra
        "Kubeflow", "Ray", "Triton Inference", "ONNX",
    ]
    skill_queries = [f"{skill} engineer London UK" for skill in skills]
    skill_queries += [f"{skill} data scientist UK remote" for skill in skills[:6]]

    # ── Mission-Driven / Nonprofit ─────────────────────────────────────────────
    nonprofit = [
        "Data Scientist nonprofit UK",
        "AI for Good engineer London",
        "Machine Learning charity UK",
        "Data Scientist social impact London",
        "AI Engineer healthcare UK NHS",
        "ML Engineer climate tech London",
        "AI safety researcher London UK",
        "Data Scientist public sector UK",
    ]

    return SearchPlan(
        primary_queries=primary,
        startup_queries=startup,
        skill_specific_queries=skill_queries,
        nonprofit_queries=nonprofit,
        sources_to_query=[
            "adzuna",
            "reed",
            "wellfound",
            "linkedin_proxy",
            "indeed_proxy",
            "uk_gov_find_a_job",  # DWP Find a Job — best-effort, see connector docstring
            "hn_who_is_hiring",   # Monthly "Ask HN: Who is hiring?" thread
            "career_pages",
            "ats_direct",         # Greenhouse + Lever + Ashby
            "funding_news",       # Newly-funded startup discovery
            "recruiter_boards",   # UK AI/ML specialist recruiters
        ],
    )
