"""
JobForge AI — Wellfound (formerly AngelList) Connector.

Primary startup job source. Uses Tavily web search as a proxy
since Wellfound's API is limited. Parses startup-specific metadata.
"""

from __future__ import annotations

import structlog

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector, detect_startup
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore


class WellfoundConnector(JobSourceConnector):
    """Wellfound/AngelList startup job connector (via Tavily search proxy)."""

    source_name = "wellfound"
    daily_quota = 30  # Conservative — Tavily has its own limits

    def __init__(self) -> None:
        self.tavily_key = settings.sources.tavily_api_key
        if TavilyClient and self.tavily_key:
            self.client = TavilyClient(api_key=self.tavily_key)
        else:
            self.client = None

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 10,
    ) -> list[RawJob]:
        if not self.client:
            logger.warning("wellfound.no_client", reason="Tavily API key not configured")
            return []

        all_jobs: list[RawJob] = []

        for query in queries:
            try:
                # Search Wellfound specifically via Tavily
                search_query = f"site:wellfound.com {query} {location}"
                results = self.client.search(
                    query=search_query,
                    max_results=min(max_results_per_query, 10),
                    include_answer=False,
                )

                for i, result in enumerate(results.get("results", [])):
                    try:
                        job = self._parse_tavily_result(result, query, i)
                        if job:
                            all_jobs.append(job)
                    except (KeyError, ValueError) as e:
                        logger.debug("wellfound.parse.skip", error=str(e))

            except Exception as e:
                logger.error("wellfound.search.error", query=query, error=str(e))

        logger.info("wellfound.search.complete", total=len(all_jobs))
        return all_jobs

    def _parse_tavily_result(self, result: dict, query: str, idx: int) -> RawJob | None:
        """Parse a Tavily search result from Wellfound pages."""
        url = result.get("url", "")
        if "wellfound.com" not in url:
            return None

        title = result.get("title", "").strip()
        content = result.get("content", "")

        # Try to extract company from URL pattern: wellfound.com/company/[name]/jobs/...
        company = "Unknown Startup"
        if "/company/" in url:
            parts = url.split("/company/")
            if len(parts) > 1:
                company_slug = parts[1].split("/")[0]
                company = company_slug.replace("-", " ").title()

        return RawJob(
            job_id=f"wellfound_{hash(url) & 0xFFFFFFFF}",
            title=title if title else query,
            company=company,
            location="UK",
            description=content[:8000],
            url=url,
            source="wellfound",
            work_model="unknown",
            is_startup=True,
            company_stage="unknown",
        )
