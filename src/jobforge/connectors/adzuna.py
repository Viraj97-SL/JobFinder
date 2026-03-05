"""
JobForge AI — Adzuna API Connector.

Primary UK job source. Free tier: 250 requests/day.
API docs: https://developer.adzuna.com/docs/search

Adzuna is UK-native and provides the best structured salary + location data.
"""

from __future__ import annotations

from datetime import date, datetime

import httpx
import structlog

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search"


class AdzunaConnector(JobSourceConnector):
    """Adzuna UK job search API connector."""

    source_name = "adzuna"
    daily_quota = 200

    def __init__(self) -> None:
        self.app_id = settings.sources.adzuna_app_id
        self.app_key = settings.sources.adzuna_app_key
        self._request_count = 0

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        all_jobs: list[RawJob] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries:
                if self._request_count >= self.daily_quota:
                    logger.warning("adzuna.quota.reached", count=self._request_count)
                    break

                try:
                    jobs = await self._search_single(client, query, max_results_per_query)
                    all_jobs.extend(jobs)
                    self._request_count += 1
                except httpx.HTTPStatusError as e:
                    logger.error("adzuna.api.error", query=query, status=e.response.status_code)
                except httpx.TimeoutException:
                    logger.error("adzuna.api.timeout", query=query)

        return all_jobs

    async def _search_single(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> list[RawJob]:
        """Execute a single Adzuna search query."""

        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": min(max_results, 50),
            "what": query,
            "where": "United Kingdom",
            "content-type": "application/json",
            "sort_by": "date",
            "max_days_old": 7,         # Only last 7 days
            "category": "it-jobs",     # IT/Tech category
        }

        response = await client.get(f"{ADZUNA_BASE_URL}/1", params=params)
        response.raise_for_status()
        data = response.json()

        jobs: list[RawJob] = []
        for result in data.get("results", []):
            try:
                job = self._parse_result(result)
                jobs.append(job)
            except (KeyError, ValueError) as e:
                logger.debug("adzuna.parse.skip", error=str(e), title=result.get("title", "?"))

        logger.info("adzuna.query.complete", query=query, results=len(jobs))
        return jobs

    def _parse_result(self, result: dict) -> RawJob:
        """Parse a single Adzuna API result into a RawJob."""

        # Extract salary (Adzuna provides min/max when available)
        salary_min = result.get("salary_min")
        salary_max = result.get("salary_max")

        # Extract location
        location_parts = []
        loc = result.get("location", {})
        if display_name := loc.get("display_name"):
            location_parts.append(display_name)
        location_str = ", ".join(location_parts) if location_parts else "UK"

        # Detect work model from title + description
        desc = result.get("description", "")
        title = result.get("title", "")
        combined = f"{title} {desc}".lower()
        work_model = "unknown"
        if "remote" in combined:
            work_model = "remote"
        elif "hybrid" in combined:
            work_model = "hybrid"
        elif "on-site" in combined or "onsite" in combined or "in-office" in combined:
            work_model = "onsite"

        # Parse date
        posted_date = None
        if created := result.get("created"):
            try:
                posted_date = datetime.fromisoformat(created.replace("Z", "+00:00")).date()
            except ValueError:
                pass

        # Company info
        company_name = result.get("company", {}).get("display_name", "Unknown")

        return RawJob(
            job_id=f"adzuna_{result['id']}",
            title=title.strip(),
            company=company_name,
            location=location_str,
            salary_min=salary_min,
            salary_max=salary_max,
            description=desc[:8000],
            url=result.get("redirect_url", ""),
            source="adzuna",
            posted_date=posted_date,
            work_model=work_model,
        )
