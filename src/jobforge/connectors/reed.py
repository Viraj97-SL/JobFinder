"""
JobForge AI — Reed.co.uk API Connector.

UK-only job board. Free API: 500 requests/day.
API docs: https://www.reed.co.uk/developers/jobseeker

Reed is excellent for UK-specific roles and provides good salary data.
"""

from __future__ import annotations

import base64
from datetime import datetime

import httpx
import structlog

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

REED_BASE_URL = "https://www.reed.co.uk/api/1.0/search"


class ReedConnector(JobSourceConnector):
    """Reed.co.uk job search API connector."""

    source_name = "reed"
    daily_quota = 400

    def __init__(self) -> None:
        self.api_key = settings.sources.reed_api_key
        # Reed uses HTTP Basic Auth with API key as username, empty password
        self._auth_header = base64.b64encode(f"{self.api_key}:".encode()).decode()
        self._request_count = 0

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        all_jobs: list[RawJob] = []

        headers = {"Authorization": f"Basic {self._auth_header}"}

        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            for query in queries:
                if self._request_count >= self.daily_quota:
                    logger.warning("reed.quota.reached", count=self._request_count)
                    break

                try:
                    jobs = await self._search_single(client, query, max_results_per_query)
                    all_jobs.extend(jobs)
                    self._request_count += 1
                except httpx.HTTPStatusError as e:
                    logger.error("reed.api.error", query=query, status=e.response.status_code)
                except httpx.TimeoutException:
                    logger.error("reed.api.timeout", query=query)

        return all_jobs

    async def _search_single(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> list[RawJob]:
        params = {
            "keywords": query,
            "resultsToTake": min(max_results, 100),
            "resultsToSkip": 0,
            "distanceFromLocation": 30,     # miles from location
        }

        response = await client.get(REED_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        jobs: list[RawJob] = []
        for result in data.get("results", []):
            try:
                job = self._parse_result(result)
                jobs.append(job)
            except (KeyError, ValueError) as e:
                logger.debug("reed.parse.skip", error=str(e))

        logger.info("reed.query.complete", query=query, results=len(jobs))
        return jobs

    def _parse_result(self, result: dict) -> RawJob:
        """Parse a single Reed API result."""

        salary_min = result.get("minimumSalary")
        salary_max = result.get("maximumSalary")
        location_name = result.get("locationName", "UK")

        # Detect work model
        desc = result.get("jobDescription", "")
        title = result.get("jobTitle", "")
        combined = f"{title} {desc}".lower()
        work_model = "unknown"
        if "remote" in combined:
            work_model = "remote"
        elif "hybrid" in combined:
            work_model = "hybrid"
        elif "on-site" in combined or "onsite" in combined:
            work_model = "onsite"

        # Parse date
        posted_date = None
        if date_str := result.get("date"):
            try:
                posted_date = datetime.strptime(date_str[:10], "%d/%m/%Y").date()
            except ValueError:
                pass

        return RawJob(
            job_id=f"reed_{result['jobId']}",
            title=title.strip(),
            company=result.get("employerName", "Unknown"),
            location=location_name,
            salary_min=salary_min,
            salary_max=salary_max,
            description=desc[:8000],
            url=result.get("jobUrl", ""),
            source="reed",
            posted_date=posted_date,
            work_model=work_model,
        )
