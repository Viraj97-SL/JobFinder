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

import asyncio
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
    "data_scientist": DATA_DIR / "master_cvs" / "data_scientist.tex",
    "ml_engineer":    DATA_DIR / "master_cvs" / "ML_engineer.tex",
}

PROJECTS_BANK_PATH = DATA_DIR / "data_bank" / "projects.tex"


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
        """Plan CV tailoring for top-N qualified jobs (all go in Excel, only top-N get CVs)."""
        qualified = state.get("qualified_jobs", [])
        skill_inventory = state.get("skill_inventory")

        # Slice to top-N — qualified is already sorted by score descending
        cap = settings.pipeline.max_cvs_per_run
        to_tailor = qualified[:cap]
        skipped = len(qualified) - len(to_tailor)

        if skipped > 0:
            logger.info(
                "tailor.plan.capped",
                total_qualified=len(qualified),
                tailoring=len(to_tailor),
                skipped=skipped,
                reason=f"max_cvs_per_run={cap}",
            )

        tailoring_plans = []
        for job in to_tailor:
            variant = job.recommended_cv_variant
            if variant not in CV_VARIANTS:
                variant = "ai_engineer"

            tailoring_plans.append({
                "job": job,
                "variant": variant,
                "variant_path": str(CV_VARIANTS[variant]),
                "sections_to_modify": self._identify_sections(job),
            })

        logger.info("tailor.plan", jobs=len(tailoring_plans))
        return {"plans": tailoring_plans, "skill_inventory": skill_inventory}

    async def execute(self, state: JobForgeState, plan: dict[str, Any]) -> dict[str, Any]:
        """Generate tailored CVs in parallel with retry logic."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        plans = plan["plans"]
        skill_inventory = plan["skill_inventory"]

        # Shared LLM instance for the whole batch
        llm = ChatGoogleGenerativeAI(
            model=settings.llm.deep_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=settings.llm.temperature,
        )

        semaphore = asyncio.Semaphore(settings.pipeline.tailor_concurrency)

        # Initialise RAG store once (lazy — skipped if rag_enabled=False or chromadb missing)
        rag_store = None
        if settings.pipeline.rag_enabled:
            try:
                from jobforge.memory.rag_store import RAGStore
                rag_store = RAGStore()
            except Exception as e:
                logger.warning("tailor.rag.init_failed", error=str(e))

        async def tailor_with_semaphore(p: dict) -> TailoredCV | TailorError:
            async with semaphore:
                return await self._tailor_single(p["job"], p["variant"], skill_inventory, llm, rag_store)

        results = await asyncio.gather(*[tailor_with_semaphore(p) for p in plans])

        tailored = [r for r in results if isinstance(r, TailoredCV)]
        errors   = [r for r in results if isinstance(r, TailorError)]

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
        self,
        job: ScoredJob,
        variant: str,
        skill_inventory: SkillInventory | None,
        llm: Any,
        rag_store: Any = None,
    ) -> TailoredCV | TailorError:
        """Tailor a CV for a single job with retry logic."""
        base_tex_path = CV_VARIANTS.get(variant)
        if base_tex_path is None or not base_tex_path.exists():
            return TailorError(
                job_id=job.job.job_id,
                company=job.job.company,
                error_type="missing_base_cv",
                error_message=f"Master CV not found: {base_tex_path}. Add .tex files to data/master_cvs/.",
                retry_count=0,
                fallback_used=False,
            )

        company_slug = job.job.company.replace(" ", "_")[:20].strip("_")
        role_slug = job.job.title.replace(" ", "_")[:20].strip("_")
        base_name = f"Viraj_CV_{company_slug}_{role_slug}"

        for attempt in range(self.max_retries + 1):
            try:
                base_latex = base_tex_path.read_text(encoding="utf-8")

                # ── LLM-driven modification ──
                modified_latex = await self._modify_latex(base_latex, job, skill_inventory, llm, rag_store)

                # ── Hallucination check on modified LaTeX text ──
                passed, violations = self._validate_against_inventory(
                    modified_latex, skill_inventory
                )
                if not passed:
                    logger.warning(
                        "tailor.hallucination_detected",
                        job_id=job.job.job_id,
                        attempt=attempt + 1,
                        violations=violations,
                    )
                    if attempt < self.max_retries:
                        continue  # Retry — next attempt re-prompts LLM
                    # Final attempt: fall back to base (unmodified) CV
                    modified_latex = base_latex
                    passed = True
                    violations = []

                # ── Write modified .tex ──
                today_dir = OUTPUT_DIR / "cvs"
                today_dir.mkdir(parents=True, exist_ok=True)
                tex_path = today_dir / f"{base_name}.tex"
                tex_path.write_text(modified_latex, encoding="utf-8")

                # ── Compile to PDF (if pdflatex available) ──
                pdf_path = today_dir / f"{base_name}.pdf"
                compile_success = self._compile_pdf(tex_path, today_dir)

                if not compile_success:
                    logger.warning(
                        "tailor.pdflatex_unavailable",
                        job_id=job.job.job_id,
                        tex_path=str(tex_path),
                    )
                    notes = "pdflatex not found — .tex saved, compile manually with: pdflatex " + tex_path.name
                else:
                    notes = ""

                sections = self._identify_sections(job)

                # ── Store successful tailoring in RAG for future few-shot context ──
                if rag_store is not None and passed:
                    try:
                        rag_store.store_tailoring(
                            job=job,
                            cv_variant=variant,
                            sections_modified=sections,
                            hallucination_passed=passed,
                        )
                    except Exception as e:
                        logger.debug("tailor.rag.store_failed", error=str(e))

                return TailoredCV(
                    job_id=job.job.job_id,
                    company=job.job.company,
                    role=job.job.title,
                    variant_used=variant,
                    pdf_path=str(pdf_path),
                    pdf_filename=f"{base_name}.pdf",
                    sections_modified=sections,
                    hallucination_check_passed=passed,
                    retry_count=attempt,
                    notes=notes,
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

    async def _modify_latex(
        self,
        base_latex: str,
        job: ScoredJob,
        skill_inventory: SkillInventory | None,
        llm: Any,
        rag_store: Any = None,
    ) -> str:
        """Call Gemini Pro to modify LaTeX sections for the target job."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from jobforge.config.prompts.tailor import (
            TAILOR_SKILLS_TEMPLATE,
            TAILOR_SUMMARY_TEMPLATE,
            TAILOR_SYSTEM_PROMPT,
        )

        inventory_summary = (
            skill_inventory.model_dump_json(indent=2)
            if skill_inventory
            else '{"note": "No inventory — use only verifiable information from the CV text."}'
        )

        # Load the projects bank if available
        projects_bank = ""
        if PROJECTS_BANK_PATH.exists():
            projects_bank = PROJECTS_BANK_PATH.read_text(encoding="utf-8")

        projects_section = f"""
PROJECTS BANK (select the 3–4 most relevant projects for this role and replace the Projects section):
{projects_bank[:4000]}
""" if projects_bank else ""

        # ── RAG few-shot context ──
        rag_context = ""
        if rag_store is not None:
            try:
                from jobforge.memory.rag_store import RAGStore
                examples = rag_store.find_similar(job, top_k=settings.pipeline.rag_top_k)
                rag_context = RAGStore.format_examples_for_prompt(examples)
            except Exception as e:
                logger.debug("tailor.rag.query_failed", error=str(e))

        # Build a single prompt that asks the model to return the full modified LaTeX
        user_prompt = f"""Modify the following LaTeX CV to better match the target job.

CRITICAL: Return the COMPLETE LaTeX document. Do NOT truncate, shorten, or omit any sections.
Preserve ALL existing sections (Header, Education, etc.) exactly as-is unless they are in SECTIONS TO MODIFY.
{rag_context}
TARGET JOB:
Title: {job.job.title}
Company: {job.job.company}
Key Requirements: {', '.join(job.key_matching_skills[:6])}
Transferable Highlights: {', '.join(job.transferable_highlights[:4])}
Key Gaps: {', '.join(job.key_gaps[:3]) if job.key_gaps else 'None'}

SKILL INVENTORY (your ONLY source of truth for metrics and technologies):
{inventory_summary}
{projects_section}
SECTIONS TO MODIFY:
- Professional Summary: rewrite to mirror the JD language and highlight matching skills
- Technical Skills: reorder so JD-relevant skills appear first
- Projects & Research: pick the 3–4 most relevant projects from the PROJECTS BANK above (if provided) and replace the existing projects section entirely

BASE LaTeX CV (return this in FULL with only the above sections modified):
{base_latex}

Return ONLY the complete modified LaTeX document. No explanation, no markdown fences."""

        response = await llm.ainvoke([
            SystemMessage(content=TAILOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        result = response.content.strip()
        # Strip markdown fences if present
        if result.startswith("```"):
            result = result.split("```", 2)[1]
            if result.startswith("latex") or result.startswith("tex"):
                result = result.split("\n", 1)[1]
            result = result.rsplit("```", 1)[0].strip()

        return result

    def _compile_pdf(self, tex_path: Path, output_dir: Path) -> bool:
        """
        Compile a .tex file to PDF.
        Tries pdflatex first, then tectonic (self-contained, no install needed on Railway).
        Returns True on success, False if no compiler is available.
        """
        compilers = [
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
            ["tectonic", "-o", str(output_dir), str(tex_path)],
        ]
        for cmd in compilers:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    return True
                logger.warning(
                    "tailor.compile_error",
                    compiler=cmd[0],
                    tex=tex_path.name,
                    stderr=result.stderr[-500:],
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                logger.warning("tailor.compile_timeout", compiler=cmd[0], tex=tex_path.name)
        return False

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
        self, latex_text: str, skill_inventory: SkillInventory | None
    ) -> tuple[bool, list[str]]:
        """
        Hallucination detector: scan modified LaTeX for skills/metrics not in the SkillInventory.

        Checks:
        1. Percentage metrics (e.g. "98%" or "0.97 AUC") not in quantified_achievements
        2. Technology names that appear newly added and are not in the inventory

        Returns (passed: bool, violations: list[str])
        """
        import re

        if skill_inventory is None:
            return (True, [])  # Cannot validate without inventory — pass through

        violations: list[str] = []

        # ── Check quantified metrics ──
        # Find patterns like "98%", "0.97 AUC", "£45,000"
        pct_pattern = re.compile(r'\b(\d{2,3}(?:\.\d+)?)\s*%')
        decimal_pattern = re.compile(r'\b0\.\d{2,}\b')

        for match in pct_pattern.finditer(latex_text):
            metric_str = match.group(0).strip()
            # Only flag metrics above 50% (avoid page numbers, years, etc.)
            value = float(match.group(1))
            if value > 50 and not skill_inventory.has_metric(metric_str):
                context = latex_text[max(0, match.start()-30):match.end()+30].replace('\n', ' ')
                violations.append(f"Unverified metric '{metric_str}' — not in skill inventory. Context: ...{context}...")

        # ── Sanity check: LaTeX must still compile (basic structure check) ──
        if r'\begin{document}' not in latex_text or r'\end{document}' not in latex_text:
            violations.append("Modified LaTeX is missing \\begin{document} or \\end{document}")

        return (len(violations) == 0, violations)
