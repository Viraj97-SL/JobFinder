"""
JobForge AI — CV & Skill Inventory Models.

The SkillInventory is the Tailor Agent's immutable ground truth.
It is extracted once from the master CVs and never modified by the LLM.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectEntry(BaseModel):
    """A project from the master CV."""
    name: str
    short_description: str
    technologies: list[str]
    quantified_metrics: list[str] = Field(default_factory=list)
    domain: str = ""
    is_deployed: bool = False
    github_url: str = ""


class WorkEntry(BaseModel):
    """A work experience entry."""
    role: str
    company: str
    location: str
    date_range: str
    achievements: list[str]
    quantified_metrics: list[str] = Field(default_factory=list)


class EduEntry(BaseModel):
    """An education entry."""
    degree: str
    institution: str
    date: str
    key_modules: list[str] = Field(default_factory=list)


class SkillInventory(BaseModel):
    """
    The COMPLETE ground truth of the user's skills and experience.

    Extracted from all three master CVs at pipeline initialisation.
    The Tailor Agent can ONLY rephrase/reorder content from this inventory.
    Any output that references skills/metrics NOT in this inventory
    triggers the hallucination detector.
    """

    technical_skills: dict[str, list[str]] = Field(
        description="Category -> list of skills, e.g. 'agent_orchestration': ['LangGraph', 'LangChain']"
    )
    projects: list[ProjectEntry]
    work_experience: list[WorkEntry]
    education: list[EduEntry]
    certifications: list[str]
    quantified_achievements: list[str] = Field(
        description="All numeric claims: '95% OTD', '0.96 AUC-ROC', '97% utilisation', etc."
    )
    volunteering: list[str] = Field(default_factory=list)

    def has_skill(self, skill: str) -> bool:
        """Check if a skill exists anywhere in the inventory (case-insensitive)."""
        skill_lower = skill.lower()
        for category_skills in self.technical_skills.values():
            if any(skill_lower in s.lower() for s in category_skills):
                return True
        return False

    def has_metric(self, metric: str) -> bool:
        """Check if a quantified achievement exists (fuzzy match)."""
        metric_lower = metric.lower()
        return any(metric_lower in a.lower() for a in self.quantified_achievements)

    def get_all_skills_flat(self) -> list[str]:
        """Return all skills as a flat list for embedding / matching."""
        return [s for skills in self.technical_skills.values() for s in skills]

    def get_project_names(self) -> list[str]:
        return [p.name for p in self.projects]

    def get_company_names(self) -> list[str]:
        return [w.company for w in self.work_experience]


class TailoredCV(BaseModel):
    """Metadata about a tailored CV generated for a specific job."""
    job_id: str
    company: str
    role: str
    variant_used: str
    pdf_path: str
    pdf_filename: str
    sections_modified: list[str] = Field(default_factory=list)
    hallucination_check_passed: bool = True
    retry_count: int = 0
    notes: str = ""


class TailorError(BaseModel):
    """Logged when CV tailoring fails after all retries."""
    job_id: str
    company: str
    error_type: str
    error_message: str
    retry_count: int
    fallback_used: bool = True
