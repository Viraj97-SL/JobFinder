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

import asyncio
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
        """
        Score jobs using a two-pass architecture:
          Pass 1 (ML gate)  — dense + sparse + exact signals filter obvious non-matches
          Pass 2 (LLM)      — Gemini Flash scores only what passed the gate, in parallel
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        from jobforge.memory.dedup_store import ScoreCache
        from jobforge.ml.prescreen import MLPrescreen

        jobs: list[RawJob] = plan["jobs"]
        skill_inventory = plan["skill_inventory"]

        if not jobs:
            return {"scored_jobs": [], "qualified_jobs": [], "prescreened_count": 0}

        # ── Pass 1: ML pre-screen gate ──
        to_llm: list[RawJob] = jobs
        if settings.pipeline.ml_prescreen_enabled and skill_inventory is not None:
            screen = MLPrescreen(skill_inventory, threshold=settings.pipeline.ml_prescreen_threshold)
            screen.fit_bm25(jobs)  # one vectorised BM25 fit across all jobs

            to_llm = []
            for job in jobs:
                passed, breakdown = screen.should_llm(job)
                if passed:
                    to_llm.append(job)
                else:
                    logger.debug(
                        "matchmaker.prescreen.filtered",
                        job_id=job.job_id,
                        title=job.title,
                        **breakdown,
                    )

            logger.info(
                "matchmaker.prescreen.done",
                total=len(jobs),
                passed=len(to_llm),
                filtered=len(jobs) - len(to_llm),
                filter_rate=f"{(len(jobs) - len(to_llm)) / len(jobs):.0%}",
            )

        # ── Pass 2: LLM scoring (parallel, with score cache) ──
        llm = ChatGoogleGenerativeAI(
            model=settings.llm.fast_model,
            google_api_key=settings.llm.gemini_api_key,
            temperature=settings.llm.temperature,
            request_timeout=45,   # Gemini Flash should respond well within 45s
        )
        cache = ScoreCache()
        cache.evict_expired()
        semaphore = asyncio.Semaphore(settings.pipeline.matchmaker_concurrency)

        async def score_with_semaphore(job: RawJob) -> ScoredJob | None:
            async with semaphore:
                try:
                    result = await self._score_job(job, skill_inventory, llm, cache)
                    return self._apply_visa_adjustments(result) if result else None
                except Exception as e:
                    logger.error("matchmaker.score.error", job_id=job.job_id, error=str(e))
                    return None

        results = await asyncio.gather(*[score_with_semaphore(j) for j in to_llm])
        cache.close()

        scored: list[ScoredJob] = [r for r in results if r is not None]
        qualified = [s for s in scored if s.overall_score >= self.threshold]
        qualified.sort(key=lambda s: s.overall_score, reverse=True)

        return {
            "scored_jobs": scored,
            "qualified_jobs": qualified,
            "prescreened_count": len(to_llm),
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

        def _bucket(score: float) -> str:
            if score >= 90: return "90-100"
            if score >= 80: return "80-89"
            if score >= 70: return "70-79"
            if score >= 60: return "60-69"
            return "below_60"

        dist: dict[str, int] = {"90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "below_60": 0}
        for s in scored:
            dist[_bucket(s.overall_score)] += 1

        summary = MatchSummary(
            total_scraped=len(state.get("raw_jobs", [])),
            total_after_dedup=len(state.get("deduped_jobs", [])),
            total_prescreened=result.get("prescreened_count", len(scored)),
            total_scored=len(scored),
            total_qualified=len(qualified),
            average_score=round(sum(scores) / len(scores), 1) if scores else 0,
            highest_score=max(scores) if scores else 0,
            highest_score_company=qualified[0].job.company if qualified else "",
            score_distribution=dist,
            sponsoring_jobs_count=sum(1 for s in qualified if s.job.offers_sponsorship),
            startup_jobs_count=sum(1 for s in qualified if s.job.is_startup),
        )

        return {
            "scored_jobs": scored,
            "qualified_jobs": qualified,
            "match_summary": summary,
        }

    # ── PRIVATE ──

    async def _score_job(
        self,
        job: RawJob,
        skill_inventory: Any,
        llm: Any,
        cache: Any,
    ) -> ScoredJob | None:
        """Score a single job. Checks cache first; falls back to Gemini Flash."""
        import json as _json

        from langchain_core.messages import HumanMessage, SystemMessage

        # ── Cache hit: skip LLM entirely ──
        cached = cache.get(job.dedup_hash)
        if cached:
            logger.debug("matchmaker.score.cache_hit", job_id=job.job_id, title=job.title)
            return ScoredJob(job=job, **cached)

        # ── Cache miss: call LLM ──
        if skill_inventory is not None:
            inventory_json = skill_inventory.model_dump_json(indent=2)
        else:
            inventory_json = '{"note": "No skill inventory loaded — score based on job description alone."}'

        user_content = MATCHMAKER_USER_TEMPLATE.format(
            title=job.title,
            company=job.company,
            location=job.location,
            salary=job.salary_display,
            description=job.description[:4000],
            skill_inventory_json=inventory_json,
        )

        response = await llm.ainvoke([
            SystemMessage(content=MATCHMAKER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()

        match_score = MatchScore.model_validate(_json.loads(raw))

        score_data = {
            "overall_score":          match_score.overall_score,
            "technical_skills_score": match_score.technical_skills_score,
            "domain_experience_score": match_score.domain_experience_score,
            "seniority_fit_score":    match_score.seniority_fit_score,
            "location_score":         match_score.location_score,
            "visa_score":             match_score.visa_score,
            "role_alignment_score":   match_score.role_alignment_score,
            "reasoning":              match_score.reasoning,
            "key_matching_skills":    match_score.key_matching_skills,
            "key_gaps":               match_score.key_gaps,
            "transferable_highlights": match_score.transferable_highlights,
            "recommended_cv_variant": match_score.recommended_cv_variant,
        }

        # Persist to cache for future runs
        cache.set(job.dedup_hash, score_data)

        logger.debug(
            "matchmaker.score.done",
            job_id=job.job_id,
            title=job.title,
            overall=match_score.overall_score,
        )

        return ScoredJob(job=job, **score_data)

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
