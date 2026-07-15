"""
JobForge AI — Scout Agent (Deep Agent).

Discovers fresh job postings from multiple UK sources with emphasis
on startups, applies visa signal detection, and deduplicates across runs.

Deep Agent Capabilities:
- PLAN:    Generates source-specific search queries
- EXECUTE: Fan-out parallel search across all connectors in two phases
- REFLECT: Evaluates source yield quality, deprioritises low-signal sources
- MEMORY:  PostgreSQL/SQLite dedup store persists across runs

Connector inventory:
  Phase 1 (parallel): adzuna, reed, wellfound, linkedin_proxy, indeed_proxy,
                       uk_gov_find_a_job (DWP Find a Job), hn_who_is_hiring
                       (HN "Ask HN: Who is hiring?" thread),
                       ats_direct (Greenhouse/Lever/Ashby), funding_news,
                       recruiter_boards, career_pages (base watchlist)
  Phase 2 (serial):   career_pages re-run with funding_news discovered companies
                       injected into the watchlist for this run
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from jobforge.agents.base import DeepAgent
from jobforge.agents.scout.planner import SearchPlan, build_search_plan
from jobforge.config.settings import settings
from jobforge.connectors.adzuna import AdzunaConnector
from jobforge.connectors.base import JobSourceConnector
from jobforge.connectors.career_pages import CareerPagesConnector
from jobforge.connectors.funding_news import FundingNewsConnector
from jobforge.connectors.greenhouse_lever import GreenhouseLeverConnector
from jobforge.connectors.hn_who_is_hiring import HNWhoIsHiringConnector
from jobforge.connectors.indeed_proxy import IndeedProxyConnector
from jobforge.connectors.linkedin_proxy import LinkedInProxyConnector
from jobforge.connectors.recruiter_boards import RecruiterBoardsConnector
from jobforge.connectors.reed import ReedConnector
from jobforge.connectors.sponsor_register import SponsorRegisterMatcher, download_sponsor_register
from jobforge.connectors.uk_gov_find_a_job import UkGovFindAJobConnector
from jobforge.connectors.wellfound import WellfoundConnector
from jobforge.memory.dedup_store import AnalyticsStore, DedupStore
from jobforge.models.job import RawJob
from jobforge.models.state import JobForgeState, ScoutMetadata
from jobforge.utils.salary_parser import detect_salary_period

logger = structlog.get_logger(__name__)


class ScoutAgent(DeepAgent):
    """
    Deep Agent that discovers jobs from multiple UK sources.

    Two-phase fan-out architecture:
      Phase 1: All connectors run in parallel (standard sources + new deep sources)
      Phase 2: CareerPages re-runs with companies discovered by FundingNewsConnector
               added to the watchlist for this run only (not persisted)
    Results are merged, enriched with visa/startup signals, and deduplicated.
    """

    name = "scout"

    def __init__(self, watchlist: list[dict] | None = None) -> None:
        """
        Args:
            watchlist: Startup career page URLs from startup_watchlist.yaml
        """
        self.watchlist = watchlist or []

        # Keep references to connectors that have side-channel data
        self._funding_connector = FundingNewsConnector()
        self._career_connector = CareerPagesConnector(watchlist=self.watchlist)
        self._ats_connector = GreenhouseLeverConnector()

        # ── Connector Registry ─────────────────────────────────────────────────
        # All connectors implement JobSourceConnector.
        # Add new sources here — no other code changes required.
        self.connectors: dict[str, JobSourceConnector] = {
            # ── Established job boards ──
            "adzuna":          AdzunaConnector(),
            "reed":            ReedConnector(),
            "wellfound":       WellfoundConnector(),
            "linkedin_proxy":  LinkedInProxyConnector(),
            "indeed_proxy":    IndeedProxyConnector(),
            # DWP "Find a Job" — best-effort scraper, see connector docstring
            "uk_gov_find_a_job": UkGovFindAJobConnector(),
            "hn_who_is_hiring":  HNWhoIsHiringConnector(),  # Monthly "who is hiring" HN thread
            # ── Deep discovery sources ──
            "ats_direct":       self._ats_connector,         # Greenhouse/Lever/Ashby JSON APIs
            "funding_news":     self._funding_connector,     # Newly-funded UK AI startups
            "recruiter_boards": RecruiterBoardsConnector(),  # Harnham, Empiric, Cord, Otta
            "career_pages":     self._career_connector,      # Curated watchlist deep crawl
        }

    # ── DEEP AGENT LIFECYCLE ───────────────────────────────────────────────────

    async def plan(self, state: JobForgeState) -> dict[str, Any]:
        """Build search plan: role-specific queries × source-specific strategies."""
        plan = build_search_plan()

        # Only activate connectors that are registered
        plan.sources_to_query = [
            s for s in plan.sources_to_query if s in self.connectors
        ]

        logger.info(
            "scout.plan.ready",
            total_queries=len(plan.all_queries),
            active_sources=plan.sources_to_query,
        )

        return {
            "plan": plan,
            "active_sources": plan.sources_to_query,
        }

    async def execute(self, state: JobForgeState, plan_data: dict[str, Any]) -> dict[str, Any]:
        """
        Two-phase fan-out:
          Phase 1: All connectors in parallel
          Phase 2: CareerPages re-run with funding-discovered companies added
        """
        plan: SearchPlan = plan_data["plan"]
        active_sources: list[str] = plan_data["active_sources"]
        start_time = time.time()

        # ── Phase 1: Parallel Fan-Out ──────────────────────────────────────────
        phase1_sources = active_sources  # All connectors run in parallel

        tasks = [
            self._search_source(self.connectors[src], plan.for_source(src))
            for src in phase1_sources
        ]
        phase1_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs: list[RawJob] = []
        source_counts: dict[str, int] = {}
        source_errors: dict[str, str] = {}

        for source_name, result in zip(phase1_sources, phase1_results):
            if isinstance(result, Exception):
                source_errors[source_name] = str(result)
                source_counts[source_name] = 0
                logger.error("scout.source.failed", source=source_name, error=str(result))
            else:
                source_counts[source_name] = len(result)
                all_jobs.extend(result)

        # ── Phase 2: Enrich CareerPages with funding discoveries ───────────────
        # FundingNewsConnector populates self._funding_connector.discovered_startups
        # after its search() call. Inject these into a fresh CareerPages run.
        new_from_funding = self._funding_connector.discovered_startups
        if new_from_funding:
            logger.info(
                "scout.phase2.funding_injection",
                new_companies=len(new_from_funding),
                names=[c["company"] for c in new_from_funding[:5]],
            )
            # Avoid re-crawling companies already in the base watchlist
            existing_urls = {e["careers_url"] for e in self.watchlist}
            truly_new = [c for c in new_from_funding if c["careers_url"] not in existing_urls]

            if truly_new:
                dynamic_crawler = CareerPagesConnector(watchlist=truly_new)
                try:
                    extra_jobs = await dynamic_crawler.search(
                        queries=plan.for_source("career_pages"),
                        location="UK",
                    )
                    source_counts["career_pages_dynamic"] = len(extra_jobs)
                    all_jobs.extend(extra_jobs)
                    logger.info(
                        "scout.phase2.dynamic_crawl.complete",
                        extra_jobs=len(extra_jobs),
                    )
                except Exception as e:
                    logger.error("scout.phase2.dynamic_crawl.error", error=str(e))

        # ── Phase 2b: Inject ATS tokens discovered via CareerPages ────────────
        # CareerPagesConnector detects embedded ATS and records the tokens.
        # These supplement the hardcoded lists in GreenhouseLeverConnector.
        ats_discovered = self._career_connector.detected_ats
        if any(ats_discovered.values()):
            logger.info("scout.ats_discovered_via_career_pages", tokens=ats_discovered)
            # These tokens are logged for future manual addition to greenhouse_lever.py
            # (auto-fetching them here would duplicate ats_direct results)

        # ── Salary Period Detection ─────────────────────────────────────────────
        # UK boards mix annual salary, day rate, and hourly rate in the same
        # salary_min/salary_max fields with no period marker. Detect it here so
        # analytics can normalise to an annual-equivalent value downstream and
        # never average a contractor day rate into the annual salary median.
        all_jobs = [
            job.model_copy(
                update={
                    "salary_period": detect_salary_period(
                        f"{job.title} {job.description}", job.salary_min, job.salary_max
                    )
                }
            )
            for job in all_jobs
        ]

        # ── Sponsor Register Cross-Check (2.4) ──────────────────────────────────
        # "JD mentions sponsorship" (NLP guess, above) vs "employer legally holds
        # a sponsor licence" (verified against the Home Office register) are
        # reported as distinct signals. Soft-fails: a download/parse error here
        # must never take down the whole scrape.
        if settings.pipeline.sponsor_register_enabled and all_jobs:
            all_jobs = await self._apply_sponsor_register(all_jobs)

        # ── Deduplication ──────────────────────────────────────────────────────
        dedup_store = DedupStore()
        try:
            # Intra-run deduplication (cross-source)
            seen_hashes: set[str] = set()
            intra_deduped: list[RawJob] = []
            for job in all_jobs:
                if job.dedup_hash not in seen_hashes:
                    seen_hashes.add(job.dedup_hash)
                    intra_deduped.append(job)

            # Cross-run deduplication (via DB)
            new_jobs = dedup_store.filter_new(intra_deduped)
        finally:
            dedup_store.close()

        # ── Analytics logging ──────────────────────────────────────────────────
        run_id = state.get("run_id", "unknown")
        skill_inventory = state.get("skill_inventory")
        analytics_store = AnalyticsStore()
        try:
            for job in new_jobs:
                try:
                    analytics_store.log_job(job, run_id, skill_inventory)
                except Exception as e:
                    logger.debug("scout.analytics.log_failed", job_id=job.job_id, error=str(e))
        finally:
            analytics_store.close()

        duration = time.time() - start_time

        metadata = ScoutMetadata(
            sources_queried=active_sources,
            source_counts=source_counts,
            source_errors=source_errors,
            total_raw=len(all_jobs),
            total_after_dedup=len(new_jobs),
            queries_used=plan.all_queries[:20],
            duration_seconds=round(duration, 2),
        )

        logger.info(
            "scout.execute.complete",
            total_raw=len(all_jobs),
            intra_dedup=len(intra_deduped),
            cross_run_new=len(new_jobs),
            duration=round(duration, 2),
            source_breakdown={k: v for k, v in source_counts.items() if v > 0},
        )

        return {
            "raw_jobs": all_jobs,
            "deduped_jobs": new_jobs,
            "metadata": metadata,
        }

    async def reflect(self, state: JobForgeState, result: dict[str, Any]) -> dict[str, Any]:
        """
        Self-reflection: evaluate source yield quality.

        Flags:
        - Sources returning 0 results (API key issues / query tuning needed)
        - Suspiciously low total yield
        - Over-reliance on a single source (>80% from one connector)
        - Newly discovered companies from funding news (log for watchlist addition)
        """
        metadata: ScoutMetadata = result["metadata"]
        deduped = result["deduped_jobs"]

        quality = "good"
        warnings: list[str] = []

        for source, count in metadata["source_counts"].items():
            if count == 0 and source not in metadata.get("source_errors", {}):
                warnings.append(
                    f"Source '{source}' returned 0 results — check API key / queries"
                )

        if len(deduped) < 5:
            quality = "warning"
            warnings.append(
                f"Only {len(deduped)} new jobs found — check API keys and connector health"
            )

        if deduped:
            source_dist: dict[str, int] = {}
            for job in deduped:
                source_dist[job.source] = source_dist.get(job.source, 0) + 1
            dominant_count = max(source_dist.values())
            if dominant_count / len(deduped) > 0.8 and len(source_dist) > 1:
                warnings.append(
                    f"Over-reliance on single source: {max(source_dist, key=source_dist.get)}"
                )

        sponsoring = sum(1 for j in deduped if j.offers_sponsorship)
        startups = sum(1 for j in deduped if j.is_startup)
        funding_discovered = len(self._funding_connector.discovered_startups)

        # Log newly discovered companies for future watchlist additions
        if funding_discovered:
            logger.info(
                "scout.reflect.funding_discoveries",
                count=funding_discovered,
                companies=[c["company"] for c in self._funding_connector.discovered_startups],
            )

        logger.info(
            "scout.reflect",
            quality=quality,
            warnings=warnings,
            sponsoring_jobs=sponsoring,
            startup_jobs=startups,
            funding_discovered=funding_discovered,
        )

        return {
            "quality": quality,
            "warnings": warnings,
            "sponsoring_jobs": sponsoring,
            "startup_jobs": startups,
            "funding_discovered": funding_discovered,
        }

    async def output(
        self, state: JobForgeState, result: dict[str, Any], reflection: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare state update for the Matchmaker Agent."""
        return {
            "raw_jobs": result["raw_jobs"],
            "deduped_jobs": result["deduped_jobs"],
            "scout_metadata": result["metadata"],
        }

    # ── PRIVATE HELPERS ────────────────────────────────────────────────────────

    async def _search_source(
        self, connector: JobSourceConnector, queries: list[str]
    ) -> list[RawJob]:
        """Execute search on a single connector with error handling."""
        return await connector.safe_search(queries=queries, location="UK")

    async def _apply_sponsor_register(self, jobs: list[RawJob]) -> list[RawJob]:
        """
        Tag each job with employer_is_licensed_sponsor by cross-referencing
        the cached UK sponsor register. Downloads/refreshes the register once
        per run (not per job) — a failure here (no internet, GOV.UK page
        changed) is logged and skipped rather than crashing the scrape.
        """
        try:
            cache_path = await download_sponsor_register(
                max_age_days=settings.pipeline.sponsor_register_cache_days
            )
            matcher = SponsorRegisterMatcher(cache_path)
        except Exception as e:
            logger.warning("scout.sponsor_register.unavailable", error=str(e))
            return jobs

        return [
            job.model_copy(
                update={"employer_is_licensed_sponsor": matcher.is_licensed_sponsor(job.company)}
            )
            for job in jobs
        ]
