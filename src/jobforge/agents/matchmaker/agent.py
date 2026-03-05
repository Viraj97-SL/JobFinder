"""
JobForge AI — Matchmaker Agent (Deep Agent).

Dual-pass scoring engine:
  Pass 1: Fast embedding pre-screen (cosine similarity)
  Pass 2: LLM structured scoring (6 weighted dimensions)

Deep Agent Capabilities:
- PLAN:    Analyse job batch, estimate token budget
- EXECUTE: Dual-pass scoring with structured JSON output
- REFLECT: Check score distribution for anomalies, calibrate
- MEMORY:  Historical score data for calibration across runs
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from jobforge.agents.base import DeepAgent
from jobforge.config.prompts.matchmaker import MATCHMAKER_SYSTEM_PROMPT, MATCHMAKER_USER_TEMPLATE
from jobforge.config.settings import settings
from jobforge.memory.dedup_store import RunHistory
from jobforge.models.job import RawJob, ScoredJob
from jobforge.models.scoring import MatchScore, MatchSummary
from jobforge.models.state import JobForgeState

logger = structlog.get_logger(__name__)


class MatchmakerAgent(DeepAgent):
    """
    Deep Agent that scores jobs against the candidate's profile.

    Uses Gemini Flash for bulk scoring (cost-effective) with
    strict Pydantic validation of JSON responses.
    """

    name = "matchmaker"

    def __init__(self) -> None:
        self.threshold = settings.pipeline.match_threshold
        self.visa_settings = settings.visa

    async def plan(self, state: JobForgeState) -> dict[str, Any]:
        """Estimate workload and token budget."""
        jobs = state.get("deduped_jobs", [])
        skill_inventory = state.get("skill_inventory")

        estimated_tokens = len(jobs) * 2000  # ~2k tokens per scoring call
        estimated_cost = estimated_tokens * 0.075 / 1_000_000  # Flash pricing

        logger.info(
            "matchmaker.plan",
            jobs_to_score=len(jobs),
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=round(estimated_cost, 4),
        )

        return {
            "jobs": jobs,
            "skill_inventory": skill_inventory,
            "estimated_tokens": estimated_tokens,
        }

    async def execute(self, state: JobForgeState, plan: dict[str, Any]) -> dict[str, Any]:
        """Score all jobs using dual-pass architecture."""
        jobs: list[RawJob] = plan["jobs"]
        skill_inventory = plan["skill_inventory"]

        if not jobs:
            return {"scored_jobs": [], "qualified_jobs": []}

        scored: list[ScoredJob] = []

        for job in jobs:
            try:
                score = await self._score_job(job, skill_inventory)
                if score:
                    # Apply visa adjustments
                    score = self._apply_visa_adjustments(score)
                    scored.append(score)
            except Exception as e:
                logger.error("matchmaker.score.error", job_id=job.job_id, error=str(e))

        # Filter and rank
        qualified = [s for s in scored if s.overall_score >= self.threshold]
        qualified.sort(key=lambda s: s.overall_score, reverse=True)

        return {
            "scored_jobs": scored,
            "qualified_jobs": qualified,
        }

    async def reflect(self, state: JobForgeState, result: dict[str, Any]) -> dict[str, Any]:
        """Check score distribution for anomalies."""
        scored = result["scored_jobs"]
        qualified = result["qualified_jobs"]

        quality = "good"
        warnings: list[str] = []

        if scored:
            scores = [s.overall_score for s in scored]
            avg = sum(scores) / len(scores)

            # Anomaly: too many high scores (scoring too loosely)
            high_count = sum(1 for s in scores if s > 90)
            if high_count / len(scores) > 0.8:
                quality = "warning"
                warnings.append(f"{high_count}/{len(scores)} jobs scored >90% — may be scoring too loosely")

            # Anomaly: too few qualified (scoring too strictly)
            if len(qualified) == 0 and len(scored) > 10:
                quality = "warning"
                warnings.append("0 qualified jobs from 10+ scored — check threshold or scoring rubric")

            # Log for calibration
            run_history = RunHistory()
            try:
                run_id = state.get("run_id", "unknown")
                run_history.log_scores(run_id, [
                    {
                        "job_hash": s.job.dedup_hash,
                        "overall_score": s.overall_score,
                        "cv_variant": s.recommended_cv_variant,
                        "offers_sponsorship": 1 if s.job.offers_sponsorship else 0,
                        "is_startup": 1 if s.job.is_startup else 0,
                    }
                    for s in scored
                ])
            finally:
                run_history.close()

        logger.info(
            "matchmaker.reflect",
            quality=quality,
            total_scored=len(scored),
            total_qualified=len(qualified),
            warnings=warnings,
        )

        return {"quality": quality, "warnings": warnings}

    async def output(
        self, state: JobForgeState, result: dict[str, Any], reflection: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare state update for the Tailor Agent."""
        scored = result["scored_jobs"]
        qualified = result["qualified_jobs"]

        # Build summary
        scores = [s.overall_score for s in scored] if scored else [0]
        summary = MatchSummary(
            total_scraped=len(state.get("raw_jobs", [])),
            total_after_dedup=len(state.get("deduped_jobs", [])),
            total_scored=len(scored),
            total_qualified=len(qualified),
            average_score=round(sum(scores) / len(scores), 1) if scores else 0,
            highest_score=max(scores) if scores else 0,
            highest_score_company=qualified[0].job.company if qualified else "",
            sponsoring_jobs_count=sum(1 for s in qualified if s.job.offers_sponsorship),
            startup_jobs_count=sum(1 for s in qualified if s.job.is_startup),
        )

        return {
            "scored_jobs": scored,
            "qualified_jobs": qualified,
            "match_summary": summary,
        }

    # ── PRIVATE ──

    async def _score_job(self, job: RawJob, skill_inventory: Any) -> ScoredJob | None:
        """
        Score a single job using Gemini Flash with structured JSON output.

        TODO: Replace this stub with actual Gemini API call.
        The prompt templates are in config/prompts/matchmaker.py.
        """
        # ─── INTEGRATION POINT ───
        # In production, this calls:
        #   from langchain_google_genai import ChatGoogleGenerativeAI
        #   llm = ChatGoogleGenerativeAI(model=settings.llm.fast_model)
        #   response = llm.invoke([system_msg, user_msg])
        #   match_score = MatchScore.model_validate_json(response.content)
        #
        # For now, return a placeholder that demonstrates the pipeline flow.
        # Replace this method body with real LLM integration in Phase 2.

        logger.debug("matchmaker.score.stub", job_id=job.job_id, title=job.title)

        # Placeholder — remove when integrating real LLM
        return None

    def _apply_visa_adjustments(self, scored: ScoredJob) -> ScoredJob:
        """Apply PSW-specific visa score adjustments."""
        adjustment = 0

        if scored.job.offers_sponsorship and self.visa_settings.prioritise_sponsoring:
            adjustment += self.visa_settings.sponsorship_boost

        if scored.job.citizens_only:
            adjustment -= self.visa_settings.citizens_only_penalty

        # Clamp to 0-100
        scored.overall_score = max(0, min(100, scored.overall_score + adjustment))
        return scored
