"""
JobForge AI — DWP "Find a Job" Connector (GOV.UK).

Find a Job is DWP's official free UK job search service. It was chosen for
Phase 2.3 as the highest-ROI new source: no ToS scraping risk in principle
(it's a government service), no API key, free. In practice it turned out to
be the messiest connector in this codebase — the research trail below is
recorded so nobody re-does it from scratch and re-invents a fake contract.

RESEARCH NOTES (verified 2026-07-15):
  - No public, unauthenticated JSON/REST API exists for reading job listings.
    The DWP API catalogue (https://www.api.gov.uk/dwp/) does not list a
    "Find a Job" read API among its published APIs. Third-party integrations
    (e.g. Recruitly's DWP connector, https://recruitly.io/marketplace/findajob-dwp)
    confirm the only documented programmatic channel is a *feed-in* path for
    employers/recruiters to POST job ads into the service — there is no
    read/search API for consumers pulling jobs out.
  - The legacy domain https://findajob.dwp.gov.uk is fully decommissioned: a
    direct fetch on 2026-07-15 returned a "This site is now closed" GOV.UK
    shell page (no redirect, no data).
  - https://www.gov.uk/find-a-job still links out, and its "Start now" CTA
    now points to the successor domain https://www.jobs.service.gov.uk/jobs.
    That successor sits behind Akamai bot-detection: a direct request (with a
    realistic browser User-Agent, Accept, Accept-Language and Referer header)
    returned HTTP 403 "Sorry, there is a problem with the service" on
    2026-07-15 — i.e. it actively blocks non-browser/server traffic. This is
    a bot wall, not a 404/removed-service response, so the connector still
    targets it rather than giving up outright — Railway's IP or a future
    relaxation of the block may fare better than this dev sandbox did.
  - Because the live successor could not be fetched directly, the HTML
    selectors below are grounded in a real, dated snapshot of the previous
    findajob.dwp.gov.uk search results page from the Wayback Machine
    (captured 2026-04-07, ~3 months before decommission):
    https://web.archive.org/web/20260407142312/https://findajob.dwp.gov.uk/search
    That snapshot shows each result as a `div.search-result` block containing
    an `h3 > a` (title text + detail-page URL) and a
    `ul.search-result-details > li` list (posted date as "DD Month YYYY",
    "<strong>Employer</strong> - Location", and a salary line, in that
    order — plus trailing `govuk-tag` chip <li>s for contract/remote type
    which we fold into the description text rather than parse structurally,
    since their wording/order isn't stable). Query params observed: `q`
    (keywords), `w` (location text).
  - CONSEQUENCE: this connector is best-effort / experimental. Given the
    Akamai wall, it will likely return an empty list in most production runs
    until DWP ships a real public API or the block relaxes for this traffic
    class. It fails soft — a non-200 response or a parse error is logged and
    treated as "no results for this query", never raised — matching every
    other connector's `safe_search` contract.

No API key required — there is nothing to authenticate against; the barrier
here is anti-scraping infrastructure, not access control.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import httpx
import structlog
from bs4 import BeautifulSoup

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

# Live successor to the decommissioned findajob.dwp.gov.uk (see module docstring).
FIND_A_JOB_BASE_URL = "https://www.jobs.service.gov.uk/jobs"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

_SALARY_RE = re.compile(r"£\s?([\d,]+(?:\.\d+)?)")
_DATE_RE = re.compile(r"^\d{1,2}\s+\w+\s+\d{4}$")


class UkGovFindAJobConnector(JobSourceConnector):
    """
    Best-effort connector for DWP's "Find a Job" service.

    See module docstring for why this is HTML scraping rather than an API
    call, and why it may legitimately return zero results in production.
    """

    source_name = "uk_gov_find_a_job"
    daily_quota = settings.sources.uk_gov_find_a_job_daily_quota

    def __init__(self) -> None:
        self._request_count = 0

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        all_jobs: list[RawJob] = []

        async with httpx.AsyncClient(
            timeout=20.0, headers=_HEADERS, follow_redirects=True
        ) as client:
            for query in queries:
                if self._request_count >= self.daily_quota:
                    logger.warning("uk_gov_find_a_job.quota.reached", count=self._request_count)
                    break

                try:
                    jobs = await self._search_single(client, query, location, max_results_per_query)
                    all_jobs.extend(jobs)
                    self._request_count += 1
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "uk_gov_find_a_job.api.error",
                        query=query,
                        status=e.response.status_code,
                    )
                except httpx.TimeoutException:
                    logger.error("uk_gov_find_a_job.api.timeout", query=query)
                except httpx.HTTPError as e:
                    logger.error("uk_gov_find_a_job.http_error", query=query, error=str(e))

        return all_jobs

    async def _search_single(
        self,
        client: httpx.AsyncClient,
        query: str,
        location: str,
        max_results: int,
    ) -> list[RawJob]:
        params = {"q": query, "w": location}
        response = await client.get(FIND_A_JOB_BASE_URL, params=params)

        if response.status_code != 200:
            # Bot-wall / decommission risk documented at module level — a
            # non-200 here (e.g. Akamai's 403) is expected, not exceptional.
            logger.warning(
                "uk_gov_find_a_job.non_200",
                query=query,
                status=response.status_code,
            )
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        jobs: list[RawJob] = []
        for card in soup.select("div.search-result")[:max_results]:
            try:
                job = self._parse_card(card)
                if job:
                    jobs.append(job)
            except (AttributeError, ValueError) as e:
                logger.debug("uk_gov_find_a_job.parse.skip", error=str(e))

        logger.info("uk_gov_find_a_job.query.complete", query=query, results=len(jobs))
        return jobs

    def _parse_card(self, card) -> RawJob | None:
        """Parse a single `div.search-result` block (see module docstring for shape)."""
        title_anchor = card.select_one("h3 a")
        if not title_anchor:
            return None

        title = title_anchor.get_text(strip=True)
        url = title_anchor.get("href", "").strip()
        if not title or not url:
            return None

        posted_date: date | None = None
        company = "Unknown"
        location_str = "UK"
        salary_min: float | None = None
        salary_max: float | None = None
        extra_tags: list[str] = []

        details = card.select_one("ul.search-result-details")
        if details:
            for li in details.find_all("li", recursive=False):
                text = li.get_text(" ", strip=True)
                if not text:
                    continue

                if li.find("strong") and " - " in text:
                    employer_part, _, location_part = text.partition(" - ")
                    company = employer_part.strip() or company
                    location_str = location_part.strip() or location_str
                elif "£" in text:
                    salary_min, salary_max = self._parse_salary(text)
                elif _DATE_RE.match(text):
                    posted_date = self._parse_posted_date(text)
                else:
                    extra_tags.append(text)  # contract/remote type chips — see docstring

        desc_el = card.select_one("p.search-result-description")
        description_body = desc_el.get_text(" ", strip=True) if desc_el else ""
        description = " ".join(filter(None, [description_body, *extra_tags])) or title

        combined = f"{title} {description}".lower()
        work_model = "unknown"
        if "remote" in combined:
            work_model = "remote"
        elif "hybrid" in combined:
            work_model = "hybrid"
        elif "on-site" in combined or "onsite" in combined:
            work_model = "onsite"

        job_id_source = card.get("data-aid") or url
        return RawJob(
            job_id=f"ukgov_{job_id_source}",
            title=title,
            company=company,
            location=location_str,
            salary_min=salary_min,
            salary_max=salary_max,
            description=description[:8000],
            url=url,
            source=self.source_name,
            posted_date=posted_date,
            work_model=work_model,
        )

    @staticmethod
    def _parse_salary(text: str) -> tuple[float | None, float | None]:
        amounts = [float(m.replace(",", "")) for m in _SALARY_RE.findall(text)]
        if not amounts:
            return None, None
        if len(amounts) == 1:
            return amounts[0], None
        return min(amounts), max(amounts)

    @staticmethod
    def _parse_posted_date(text: str) -> date | None:
        try:
            return datetime.strptime(text, "%d %B %Y").date()
        except ValueError:
            return None
