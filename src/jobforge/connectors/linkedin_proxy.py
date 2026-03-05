"""
JobForge AI — LinkedIn Jobs Connector (via Tavily/SerpAPI proxy).

De-prioritised source. No direct API available.
Uses Tavily web search to find LinkedIn job listings indirectly.
"""

from __future__ import annotations

import structlog

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore


class LinkedInProxyConnector(JobSourceConnector):
    """LinkedIn Jobs via Tavily search proxy. Low priority, supplementary source."""

    source_name = "linkedin_proxy"
    daily_quota = 20

    def __init__(self) -> None:
        self.tavily_key = settings.sources.tavily_api_key
        self.client = TavilyClient(api_key=self.tavily_key) if TavilyClient and self.tavily_key else None

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 10,
    ) -> list[RawJob]:
        if not self.client:
            logger.warning("linkedin_proxy.no_client")
            return []

        all_jobs: list[RawJob] = []
        for query in queries[:3]:  # Limit queries — low priority source
            try:
                results = self.client.search(
                    query=f"site:linkedin.com/jobs {query} {location}",
                    max_results=min(max_results_per_query, 5),
                    include_answer=False,
                )
                for result in results.get("results", []):
                    url = result.get("url", "")
                    if "linkedin.com/jobs" not in url:
                        continue
                    all_jobs.append(RawJob(
                        job_id=f"linkedin_{hash(url) & 0xFFFFFFFF}",
                        title=result.get("title", "").strip(),
                        company="See LinkedIn",
                        location=location,
                        description=result.get("content", "")[:8000],
                        url=url,
                        source="linkedin_proxy",
                    ))
            except Exception as e:
                logger.error("linkedin_proxy.error", query=query, error=str(e))

        return all_jobs
