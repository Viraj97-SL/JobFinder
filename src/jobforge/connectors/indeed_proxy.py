"""
JobForge AI — Indeed Proxy Connector (via SerpAPI Google Jobs).

Fallback source. Highest volume but noisiest.
"""

from __future__ import annotations

import httpx
import structlog

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"


class IndeedProxyConnector(JobSourceConnector):
    """Indeed jobs via SerpAPI Google Jobs endpoint."""

    source_name = "indeed_proxy"
    daily_quota = 50

    def __init__(self) -> None:
        self.api_key = settings.sources.serpapi_key

    async def search(
        self,
        queries: list[str],
        location: str = "United Kingdom",
        max_results_per_query: int = 10,
    ) -> list[RawJob]:
        if not self.api_key:
            logger.warning("indeed_proxy.no_key")
            return []

        all_jobs: list[RawJob] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries:
                try:
                    params = {
                        "engine": "google_jobs",
                        "q": query,
                        "location": location,
                        "api_key": self.api_key,
                        "num": min(max_results_per_query, 10),
                    }
                    resp = await client.get(SERPAPI_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                    for result in data.get("jobs_results", []):
                        all_jobs.append(RawJob(
                            job_id=f"indeed_{hash(result.get('job_id', '')) & 0xFFFFFFFF}",
                            title=result.get("title", "").strip(),
                            company=result.get("company_name", "Unknown"),
                            location=result.get("location", location),
                            description=result.get("description", "")[:8000],
                            url=result.get("related_links", [{}])[0].get("link", "")
                                if result.get("related_links") else "",
                            source="indeed_proxy",
                        ))
                except Exception as e:
                    logger.error("indeed_proxy.error", query=query, error=str(e))

        return all_jobs
