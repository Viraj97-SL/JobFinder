"""
JobForge AI — Scout Agent (Deep Agent).

Discovers fresh job postings from multiple UK sources with emphasis
on startups, applies visa signal detection, and deduplicates across runs.

Deep Agent Capabilities:
- PLAN:    Generates source-specific search queries
- EXECUTE: Fan-out parallel search across all connectors
- REFLECT: Evaluates source yield quality, deprioritises low-signal sources
- MEMORY:  SQLite dedup store persists across runs
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from jobforge.agents.base import DeepAgent
from jobforge.agents.scout.planner import SearchPlan, build_search_plan
from jobforge.connectors.adzuna import AdzunaConnector
from jobforge.connectors.base import JobSourceConnector
from jobforge.connectors.career_pages import CareerPagesConnector
from jobforge.connectors.indeed_proxy import IndeedProxyConnector
from jobforge.connectors.linkedin_proxy import LinkedInProxyConnector
from jobforge.connectors.reed import ReedConnector
from jobforge.connectors.wellfound import WellfoundConnector
from jobforge.memory.dedup_store import DedupStore
from jobforge.models.job import RawJob
from jobforge.models.state import JobForgeState, ScoutMetadata

logger = structlog.get_logger(__name__)


class ScoutAgent(DeepAgent):
    """
    Deep Agent that discovers jobs from multiple UK sources.

    Fan-out architecture: all connectors execute in parallel via asyncio.gather.
    Results are merged, enriched with visa/startup signals, and deduplicated.
    """

    name = "scout"

    def __init__(self, watchlist: list[dict] | None = None) -> None:
        """
        Args:
            watchlist: Startup career page URLs from startup_watchlist.yaml
        """
        self.watchlist = watchlist or []

        # ── Connector Registry ──
        # Add/remove connectors here. Each implements JobSourceConnector.
        self.connectors: dict[str, JobSourceConnector] = {
            "adzuna": AdzunaConnector(),
            "reed": ReedConnector(),
            "wellfound": WellfoundConnector(),
            "linkedin_proxy": LinkedInProxyConnector(),
            "indeed_proxy": IndeedProxyConnector(),
            "career_pages": CareerPagesConnector(watchlist=self.watchlist),
        }

    # ── DEEP AGENT LIFECYCLE ──

    async def plan(self, state: JobForgeState) -> dict[str, Any]:
        """Build search plan: role-specific queries × source-specific strategies."""
        plan = build_search_plan()

        # Filter to only configured (non-empty API key) connectors
        active_sources = []
        for name, connector in self.connectors.items():
            if name in plan.sources_to_query:
                active_sources.append(name)

        plan.sources_to_query = active_sources

        logger.info(
            "scout.plan.ready",
            total_queries=len(plan.all_queries),
            active_sources=active_sources,
        )

        return {
            "plan": plan,
            "active_sources": active_sources,
        }

    async def execute(self, state: JobForgeState, plan_data: dict[str, Any]) -> dict[str, Any]:
        """Fan-out search across all active connectors, then merge + dedup."""
        plan: SearchPlan = plan_data["plan"]
        active_sources: list[str] = plan_data["active_sources"]
        start_time = time.time()

        # ── Fan-Out: Parallel Search ──
        tasks = []
        for source_name in active_sources:
            connector = self.connectors[source_name]
            queries = plan.for_source(source_name)
            tasks.append(self._search_source(connector, queries))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── Merge Results ──
        all_jobs: list[RawJob] = []
        source_counts: dict[str, int] = {}
        source_errors: dict[str, str] = {}

        for source_name, result in zip(active_sources, results):
            if isinstance(result, Exception):
                source_errors[source_name] = str(result)
                source_counts[source_name] = 0
                logger.error("scout.source.failed", source=source_name, error=str(result))
            else:
                source_counts[source_name] = len(result)
                all_jobs.extend(result)

        # ── Deduplication (cross-source + cross-run) ──
        dedup_store = DedupStore()
        try:
            # First: deduplicate within this run (cross-source)
            seen_hashes: set[str] = set()
            intra_deduped: list[RawJob] = []
            for job in all_jobs:
                if job.dedup_hash not in seen_hashes:
                    seen_hashes.add(job.dedup_hash)
                    intra_deduped.append(job)

            # Then: filter out jobs seen in previous runs
            new_jobs = dedup_store.filter_new(intra_deduped)
        finally:
            dedup_store.close()

        duration = time.time() - start_time

        metadata = ScoutMetadata(
            sources_queried=active_sources,
            source_counts=source_counts,
            source_errors=source_errors,
            total_raw=len(all_jobs),
            total_after_dedup=len(new_jobs),
            queries_used=plan.all_queries[:20],  # Log first 20
            duration_seconds=round(duration, 2),
        )

        logger.info(
            "scout.execute.complete",
            total_raw=len(all_jobs),
            intra_dedup=len(intra_deduped),
            cross_run_new=len(new_jobs),
            duration=round(duration, 2),
        )

        return {
            "raw_jobs": all_jobs,
            "deduped_jobs": new_jobs,
            "metadata": metadata,
        }

    async def reflect(self, state: JobForgeState, result: dict[str, Any]) -> dict[str, Any]:
        """
        Self-reflection: evaluate source yield quality.

        If a source consistently returns 0 results, it should be
        deprioritised in future runs (logged for manual review).
        """
        metadata: ScoutMetadata = result["metadata"]
        deduped = result["deduped_jobs"]

        quality = "good"
        warnings: list[str] = []

        # Check if any source returned 0 results
        for source, count in metadata["source_counts"].items():
            if count == 0 and source not in metadata["source_errors"]:
                warnings.append(f"Source '{source}' returned 0 results — may need query tuning")

        # Check if total yield is suspiciously low
        if len(deduped) < 3:
            quality = "warning"
            warnings.append(f"Only {len(deduped)} new jobs found — check API keys and queries")

        # Check if all jobs are from a single source (over-reliance)
        if deduped:
            source_dist = {}
            for job in deduped:
                source_dist[job.source] = source_dist.get(job.source, 0) + 1
            dominant = max(source_dist.values())
            if dominant / len(deduped) > 0.8 and len(source_dist) > 1:
                warnings.append(f"Over-reliance on single source: {max(source_dist, key=source_dist.get)}")

        # Check sponsorship signal coverage
        sponsoring = sum(1 for j in deduped if j.offers_sponsorship)
        startups = sum(1 for j in deduped if j.is_startup)

        logger.info(
            "scout.reflect",
            quality=quality,
            warnings=warnings,
            sponsoring_jobs=sponsoring,
            startup_jobs=startups,
        )

        return {
            "quality": quality,
            "warnings": warnings,
            "sponsoring_jobs": sponsoring,
            "startup_jobs": startups,
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

    # ── PRIVATE HELPERS ──

    async def _search_source(
        self, connector: JobSourceConnector, queries: list[str]
    ) -> list[RawJob]:
        """Execute search on a single connector with error handling."""
        return await connector.safe_search(queries=queries, location="UK")
