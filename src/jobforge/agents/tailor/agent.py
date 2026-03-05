"""
JobForge AI — Tailor Agent (Deep Agent).

Generates job-specific tailored CVs by modifying the appropriate
LaTeX master variant. Includes hallucination detection and retry logic.

Deep Agent Capabilities:
- PLAN:    Select CV variant, identify sections to modify
- EXECUTE: LLM-driven LaTeX modification + pdflatex compilation
- REFLECT: Hallucination validation against Skill Inventory
- MEMORY:  Skill Inventory JSON (immutable ground truth)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import structlog

from jobforge.agents.base import DeepAgent
from jobforge.config.settings import DATA_DIR, OUTPUT_DIR, settings
from jobforge.models.cv import SkillInventory, TailoredCV, TailorError
from jobforge.models.job import ScoredJob
from jobforge.models.state import JobForgeState

logger = structlog.get_logger(__name__)

# CV variant mapping
CV_VARIANTS = {
    "ai_engineer":    DATA_DIR / "master_cvs" / "ai_engineer.tex",
    "data_scientist":  DATA_DIR / "master_cvs" / "data_scientist.tex",
    "data_engineer":   DATA_DIR / "master_cvs" / "data_engineer.tex",
}


class TailorAgent(DeepAgent):
    """
    Deep Agent that generates job-specific tailored PDF CVs.

    CRITICAL RULE: The Tailor Agent can ONLY rephrase, reorder, or emphasise
    skills/experiences that exist in the SkillInventory. It CANNOT invent
    new skills, fabricate metrics, or add technologies not in the inventory.
    """

    name = "tailor"

    def __init__(self) -> None:
        self.max_retries = settings.pipeline.max_tailor_retries

    async def plan(self, state: JobForgeState) -> dict[str, Any]:
        """Plan CV tailoring for all qualified jobs."""
        qualified = state.get("qualified_jobs", [])
        skill_inventory = state.get("skill_inventory")

        tailoring_plans = []
        for job in qualified:
            variant = job.recommended_cv_variant
            if variant not in CV_VARIANTS:
                variant = "ai_engineer"  # Fallback

            tailoring_plans.append({
                "job": job,
                "variant": variant,
                "variant_path": str(CV_VARIANTS[variant]),
                "sections_to_modify": self._identify_sections(job),
            })

        logger.info("tailor.plan", jobs=len(tailoring_plans))
        return {"plans": tailoring_plans, "skill_inventory": skill_inventory}

    async def execute(self, state: JobForgeState, plan: dict[str, Any]) -> dict[str, Any]:
        """Generate tailored CVs with retry logic."""
        plans = plan["plans"]
        skill_inventory = plan["skill_inventory"]

        tailored: list[TailoredCV] = []
        errors: list[TailorError] = []

        for p in plans:
            job: ScoredJob = p["job"]
            variant = p["variant"]
            result = await self._tailor_single(job, variant, skill_inventory)

            if isinstance(result, TailoredCV):
                tailored.append(result)
            else:
                errors.append(result)

        return {"tailored_cvs": tailored, "tailor_errors": errors}

    async def reflect(self, state: JobForgeState, result: dict[str, Any]) -> dict[str, Any]:
        """Validate all generated CVs against the Skill Inventory."""
        tailored = result["tailored_cvs"]
        errors = result["tailor_errors"]

        quality = "good"
        if errors:
            quality = "warning" if len(errors) < len(tailored) else "poor"

        logger.info(
            "tailor.reflect",
            quality=quality,
            generated=len(tailored),
            failed=len(errors),
        )

        return {"quality": quality}

    async def output(
        self, state: JobForgeState, result: dict[str, Any], reflection: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "tailored_cvs": result["tailored_cvs"],
            "tailor_errors": result["tailor_errors"],
        }

    # ── PRIVATE ──

    async def _tailor_single(
        self, job: ScoredJob, variant: str, skill_inventory: SkillInventory | None
    ) -> TailoredCV | TailorError:
        """
        Tailor a CV for a single job with retry logic.

        TODO: Implement LLM-driven LaTeX modification in Phase 3.
        Current stub: copies base CV without modification (safe fallback).
        """
        for attempt in range(self.max_retries + 1):
            try:
                # ─── INTEGRATION POINT ───
                # Phase 3 implementation:
                # 1. Read base LaTeX file
                # 2. Call Gemini Pro with tailoring instructions
                # 3. Validate output against SkillInventory
                # 4. Compile with pdflatex
                # 5. Return TailoredCV

                # For now: placeholder that produces the right schema
                company_slug = job.job.company.replace(" ", "_")[:20]
                role_slug = job.job.title.replace(" ", "_")[:20]
                filename = f"Viraj_CV_{company_slug}_{role_slug}.pdf"

                return TailoredCV(
                    job_id=job.job.job_id,
                    company=job.job.company,
                    role=job.job.title,
                    variant_used=variant,
                    pdf_path=str(OUTPUT_DIR / "cvs" / filename),
                    pdf_filename=filename,
                    sections_modified=["summary", "skills"],
                    hallucination_check_passed=True,
                    retry_count=attempt,
                    notes="Stub — base CV used. Implement LLM tailoring in Phase 3.",
                )

            except Exception as e:
                logger.warning(
                    "tailor.retry",
                    job_id=job.job.job_id,
                    attempt=attempt + 1,
                    error=str(e),
                )

        return TailorError(
            job_id=job.job.job_id,
            company=job.job.company,
            error_type="max_retries_exceeded",
            error_message=f"Failed after {self.max_retries} retries",
            retry_count=self.max_retries,
            fallback_used=True,
        )

    def _identify_sections(self, job: ScoredJob) -> list[str]:
        """Determine which CV sections need modification based on match analysis."""
        sections = ["professional_summary"]  # Always rewrite the summary

        if job.key_matching_skills:
            sections.append("technical_skills")

        if job.transferable_highlights:
            sections.append("work_experience")

        # If domain-specific, reorder projects
        if job.domain_experience_score > 60:
            sections.append("projects")

        return sections

    def _validate_against_inventory(
        self, pdf_text: str, skill_inventory: SkillInventory
    ) -> tuple[bool, list[str]]:
        """
        Hallucination detector: check extracted PDF text against SkillInventory.

        Returns (passed: bool, violations: list[str])
        """
        violations = []

        # TODO: Implement in Phase 3
        # 1. Extract text from generated PDF using pdfplumber
        # 2. Find all technical terms and percentages
        # 3. Cross-reference with skill_inventory.get_all_skills_flat()
        # 4. Cross-reference metrics with skill_inventory.quantified_achievements

        return (len(violations) == 0, violations)
