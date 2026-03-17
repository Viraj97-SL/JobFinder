"""
JobForge AI — UK AI/ML Recruiter Boards Connector.

Targets specialist UK recruitment agencies that focus exclusively on
data science, machine learning, and AI roles. These boards surface roles
that never appear on Adzuna or Reed because they're filled through
agency-exclusive pipelines.

Agencies covered:
  - Harnham       (harnham.com)           — UK's largest AI/data recruiter
  - Empiric        (empiric.co.uk)         — tech & data specialist
  - Understanding Recruitment (ur.co.uk)  — data & analytics
  - Xcede          (xcede.com)             — data science & ML
  - Lorien         (lorienresourcing.co.uk)— tech & data
  - Otta / Rippling Jobs (otta.com)        — startup-focused, London-heavy
  - Cord           (cord.co)               — direct tech hiring

Discovery method: Tavily `site:` searches for each agency, extracting
job URLs and descriptions from search results.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

import structlog

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore


# ── Agency search targets ──────────────────────────────────────────────────────
RECRUITER_SOURCES = [
    {
        "name": "Harnham",
        "site": "harnham.com",
        "queries": [
            "site:harnham.com machine learning engineer London",
            "site:harnham.com data scientist AI London UK",
            "site:harnham.com NLP engineer UK",
        ],
    },
    {
        "name": "Empiric",
        "site": "empiric.co.uk",
        "queries": [
            "site:empiric.co.uk machine learning AI engineer London",
            "site:empiric.co.uk data scientist deep learning UK",
        ],
    },
    {
        "name": "Understanding Recruitment",
        "site": "ur.co.uk",
        "queries": [
            "site:ur.co.uk data scientist machine learning London",
            "site:ur.co.uk AI engineer UK",
        ],
    },
    {
        "name": "Xcede",
        "site": "xcede.com",
        "queries": [
            "site:xcede.com machine learning engineer London UK",
            "site:xcede.com data scientist AI London",
        ],
    },
    {
        "name": "Cord",
        "site": "cord.co",
        "queries": [
            "site:cord.co AI engineer London startup",
            "site:cord.co machine learning engineer UK",
        ],
    },
    {
        "name": "Otta",
        "site": "otta.com",
        "queries": [
            "site:otta.com AI machine learning engineer London",
            "site:otta.com data scientist startup London UK",
        ],
    },
]

# AI/ML job relevance keywords
AI_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "llm", "nlp", "computer vision", "deep learning", "pytorch", "tensorflow",
    "langchain", "langgraph", "neural", "reinforcement learning", "mlops",
    "data engineer", "research scientist", "applied scientist",
    "generative ai", "foundation model", "multimodal",
}


class RecruiterBoardsConnector(JobSourceConnector):
    """
    Searches UK AI/ML specialist recruiter job boards via Tavily.

    Surfaces agency-exclusive roles not visible on public job boards.
    Each agency query uses Tavily's `site:` operator so results are
    scoped directly to that agency's listings.
    """

    source_name = "recruiter_boards"
    daily_quota = 30

    def __init__(self) -> None:
        self.tavily_key = settings.sources.tavily_api_key
        if TavilyClient and self.tavily_key:
            self._tavily = TavilyClient(api_key=self.tavily_key)
        else:
            self._tavily = None

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        if not self._tavily:
            logger.warning("recruiter_boards.no_tavily", reason="TAVILY_API_KEY not set")
            return []

        all_jobs: list[RawJob] = []

        for agency in RECRUITER_SOURCES:
            for query in agency["queries"]:
                try:
                    results = self._tavily.search(
                        query=query,
                        max_results=8,
                        search_depth="basic",
                    )
                    for r in results.get("results", []):
                        job = self._parse_result(r, agency["name"])
                        if job and self._is_relevant(job):
                            all_jobs.append(job)
                except Exception as e:
                    logger.warning(
                        "recruiter_boards.query.error",
                        agency=agency["name"],
                        error=str(e),
                    )

        logger.info("recruiter_boards.complete", total=len(all_jobs))
        return all_jobs

    def _parse_result(self, result: dict, agency: str) -> RawJob | None:
        url = result.get("url", "")
        title = result.get("title", "").strip()
        content = result.get("content", "")

        if not url or not title:
            return None

        # Try to extract job title from page title (often "Job Title - Company | Agency")
        clean_title = re.split(r"\s*[|\-—]\s*", title)[0].strip()
        if not clean_title or len(clean_title) < 5:
            clean_title = title

        # Extract location from content if mentioned
        location = "London, UK"
        loc_match = re.search(
            r"\b(London|Manchester|Edinburgh|Remote|Hybrid|UK\s+Remote)\b",
            content, re.IGNORECASE
        )
        if loc_match:
            location = loc_match.group(1)

        # Extract salary if present
        salary = ""
        sal_match = re.search(
            r"£[\d,]+(?:\s*[-–]\s*£[\d,]+)?(?:\s*(?:per\s+annum|pa|k))?",
            content, re.IGNORECASE
        )
        if sal_match:
            salary = sal_match.group(0)

        description = content
        if salary:
            description = f"Salary: {salary}\n\n{content}"

        return RawJob(
            job_id=f"rec_{hash(url) & 0xFFFFFFFF}",
            title=clean_title,
            company=self._extract_company(title, content, agency),
            location=location,
            description=description[:8000],
            url=url,
            source="recruiter_boards",
            is_startup=False,  # Recruiter boards include all company sizes
            company_stage="unknown",
        )

    @staticmethod
    def _extract_company(title: str, content: str, agency: str) -> str:
        """Best-effort company extraction from job title/content."""
        # Often in title as "Job Title at Company" or "Job Title - Company"
        at_match = re.search(r"\bat\s+([A-Z][A-Za-z0-9\s&]+?)(?:\s*[|\-]|$)", title)
        if at_match:
            return at_match.group(1).strip()

        dash_parts = re.split(r"\s*[|\-—]\s*", title)
        if len(dash_parts) >= 2:
            # Second part often has company name, unless it's the agency name
            candidate = dash_parts[1].strip()
            if candidate.lower() != agency.lower() and len(candidate) > 2:
                return candidate

        return f"Via {agency}"

    @staticmethod
    def _is_relevant(job: RawJob) -> bool:
        text = f"{job.title} {job.description}".lower()
        return any(kw in text for kw in AI_KEYWORDS)
