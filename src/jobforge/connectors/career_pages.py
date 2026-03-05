"""
JobForge AI — Company Career Pages Connector.

Crawls a curated list of UK AI startup career pages.
The watchlist is in data/startup_watchlist.yaml.
"""

from __future__ import annotations

import httpx
import structlog
from bs4 import BeautifulSoup

from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)


class CareerPagesConnector(JobSourceConnector):
    """Crawls curated company career pages for job listings."""

    source_name = "career_pages"
    daily_quota = 60

    def __init__(self, watchlist: list[dict] | None = None) -> None:
        """
        Args:
            watchlist: List of dicts with keys: company, careers_url, stage
                       Loaded from startup_watchlist.yaml by the Scout Agent.
        """
        self.watchlist = watchlist or []

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        """Crawl all watchlist career pages. Queries are used as keyword filters."""

        if not self.watchlist:
            logger.info("career_pages.empty_watchlist")
            return []

        all_jobs: list[RawJob] = []
        keywords = {q.lower() for q in queries}

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            for entry in self.watchlist:
                try:
                    company = entry["company"]
                    url = entry["careers_url"]
                    stage = entry.get("stage", "unknown")

                    resp = await client.get(url)
                    if resp.status_code != 200:
                        logger.debug("career_pages.fetch.skip", company=company, status=resp.status_code)
                        continue

                    soup = BeautifulSoup(resp.text, "html.parser")
                    page_text = soup.get_text(separator=" ", strip=True).lower()

                    # Check if any target keywords appear on the page
                    matching_keywords = [k for k in keywords if k in page_text]
                    if not matching_keywords:
                        continue

                    # Find job-like links on the page
                    job_links = self._extract_job_links(soup, url)

                    for link_text, link_url in job_links[:max_results_per_query]:
                        if any(k in link_text.lower() for k in keywords):
                            all_jobs.append(RawJob(
                                job_id=f"career_{hash(link_url) & 0xFFFFFFFF}",
                                title=link_text.strip(),
                                company=company,
                                location="UK",
                                description=f"Found on {company} career page. Keywords matched: {', '.join(matching_keywords)}",
                                url=link_url,
                                source="career_pages",
                                is_startup=True,
                                company_stage=stage,
                            ))

                except Exception as e:
                    logger.error("career_pages.error", company=entry.get("company"), error=str(e))

        logger.info("career_pages.complete", total=len(all_jobs))
        return all_jobs

    def _extract_job_links(self, soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
        """Extract likely job listing links from a careers page."""
        links: list[tuple[str, str]] = []
        job_keywords = {"engineer", "scientist", "analyst", "developer", "ml", "ai", "data", "machine learning"}

        for a_tag in soup.find_all("a", href=True):
            text = a_tag.get_text(strip=True)
            href = a_tag["href"]

            if not text or len(text) < 5:
                continue

            # Resolve relative URLs
            if href.startswith("/"):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)

            # Check if link text looks like a job title
            if any(kw in text.lower() for kw in job_keywords):
                links.append((text, href))

        return links
